"""
Stage 2: Professor Enrichment
Reads professors_raw.json → writes professors_enriched.json

Adds per professor:
  - prof_id          (sequential int 1, 2, 3...)
  - topic_gap        (pure Python, no LLM)
  - research_vision  (Mistral-7B via HF chat_completion)
  - method_type      (Mistral-7B via HF chat_completion)
  - future_work      (Mistral-7B via HF chat_completion)

Embeddings NOT done here — Stage 3.
"""

import json
import os
import time
from huggingface_hub import InferenceClient
import dotenv
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API"))


# HF_TOKEN    = os.getenv("HF_TOKEN")
# HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE  = "professors_raw.json"
OUTPUT_FILE = "professors_enriched.json"
HF_DELAY    = 9.0

# client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

# ── Topic Gap (no LLM) ────────────────────────────────────────────────────────

def compute_topic_gap(professor: dict) -> list[str]:
    career_topics = set(professor.get("openalex_topics") or [])
    recent_topics = set(
        topic
        for paper in (professor.get("papers") or [])[:5]
        for topic in (paper.get("topics") or [])
    )
    return sorted(recent_topics - career_topics)

# ── Abstract trimmer ──────────────────────────────────────────────────────────

def trim_abstract(abstract: str) -> str:
    """Keep first 2 + last 2 sentences — where what/future-work live."""
    sentences = [s.strip() for s in abstract.strip().split(". ") if s.strip()]
    if len(sentences) <= 4:
        return abstract
    head = ". ".join(sentences[:2])
    tail = ". ".join(sentences[-2:])
    return f"{head}. [...] {tail}."

# ── LLM call ─────────────────────────────────────────────────────────────────

def build_messages(professor: dict) -> list | None:
    papers = [
        p for p in (professor.get("papers") or [])
        if p.get("abstract", "").strip()
    ][:5]

    if not papers:
        return None

    papers_text = ""
    for i, p in enumerate(papers, 1):
        trimmed = trim_abstract(p["abstract"])
        papers_text += f"\nPaper {i} ({p.get('year')}): {p['title']}\n{trimmed}\n"

    system_msg = "You are a research analyst. Extract structured information from academic paper excerpts. Always respond with valid JSON only, no explanation, no markdown."

    user_msg = f"""Analyze these recent papers from one professor:
{papers_text}

Return ONLY this JSON object, nothing else:
{{
  "research_vision": "100 words max describing what problems they work on and current direction",
  "method_type": "one of: theoretical / empirical / systems / interdisciplinary",
  "future_work": ["specific open problem 1", "specific open problem 2", "specific open problem 3"]
}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]


# def call_llm(messages: list) -> dict | None:
#     try:
#         response = client.chat_completion(
#             messages=messages,
#             max_tokens=400,
#             temperature=0.2,
#         )
#         text = response.choices[0].message.content.strip()

#         # Strip markdown fences if model adds them
#         if "```" in text:
#             text = text.split("```")[1]
#             if text.startswith("json"):
#                 text = text[4:]
#         text = text.strip()

#         return json.loads(text)

#     except json.JSONDecodeError as e:
#         print(f"\n    [LLM] JSON parse error: {e}")
#         return None
#     except Exception as e:
#         print(f"\n    [LLM] Error: {e}")
#         return None

def call_llm(messages: list) -> dict | None:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast + free
            messages=messages,
            max_tokens=300,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text.strip())

    except json.JSONDecodeError as e:
        print(f"\n    [LLM] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"\n    [LLM] Error: {e}")
        return None

def enrich(retry_failed=False):
    with open(INPUT_FILE) as f:
        raw_profs = json.load(f)
    print(f"Loaded {len(raw_profs)} professors from {INPUT_FILE}\n")

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            enriched = json.load(f)

        if retry_failed:
            # Only skip ones that already have research_vision filled
            done_ids = {
                p["openalex_id"] for p in enriched
                if p.get("research_vision") is not None
            }
            failed_count = len(enriched) - len(done_ids)
            print(f"Retry mode — {len(done_ids)} already enriched, {failed_count} will retry")
        else:
            done_ids = {p["openalex_id"] for p in enriched}
            print(f"Resuming — {len(enriched)} already saved")
    else:
        enriched, done_ids = [], set()

    # Index for in-place updates
    enriched_index = {p["openalex_id"]: i for i, p in enumerate(enriched)}

    for raw in raw_profs:
        oa_id = raw["openalex_id"]

        if oa_id in done_ids:
            print(f"  skip → {raw['name']}")
            continue

        # Reuse existing prof_id if record already exists
        if oa_id in enriched_index:
            prof_id = enriched[enriched_index[oa_id]]["prof_id"]
        else:
            prof_id = len(enriched) + 1

        name = raw["name"]
        print(f"  [{prof_id}] {name}", end="", flush=True)

        topic_gap = compute_topic_gap(raw)
        messages  = build_messages(raw)

        llm_result = None
        if messages:
            time.sleep(HF_DELAY)
            llm_result = call_llm(messages)

        if llm_result:
            research_vision = llm_result.get("research_vision")
            method_type     = llm_result.get("method_type")
            future_work     = llm_result.get("future_work") or []
            print(f" — ✓ ({method_type})")
        else:
            research_vision = None
            method_type     = None
            future_work     = []
            print(f" — ✗ failed")

        professor = {
            "prof_id":         prof_id,
            "openalex_id":     oa_id,
            "name":            name,
            "institution":     raw.get("institution"),
            "institution_id":  raw.get("institution_id"),
            "works_count":     raw.get("works_count"),
            "cited_by_count":  raw.get("cited_by_count"),
            "openalex_topics": raw.get("openalex_topics") or [],
            "topic_gap":       topic_gap,
            "papers":          raw.get("papers") or [],
            "research_vision": research_vision,
            "method_type":     method_type,
            "future_work":     future_work,
            "embedding":       None,
        }

        # Update in-place if exists, else append
        if oa_id in enriched_index:
            enriched[enriched_index[oa_id]] = professor
        else:
            enriched.append(professor)
            enriched_index[oa_id] = len(enriched) - 1

        save(enriched)

    # Summary at end
    nulls = sum(1 for p in enriched if p.get("research_vision") is None)
    print(f"\nDone. {len(enriched)} total | {len(enriched)-nulls} enriched | {nulls} still null")




# ── Save ──────────────────────────────────────────────────────────────────────

def save(data: list):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUTPUT_FILE)

# ── Main ──────────────────────────────────────────────────────────────────────



def inspect(n=3):
    with open(OUTPUT_FILE) as f:
        profs = json.load(f)
    print(f"\nTotal enriched: {len(profs)}\n")
    for p in profs[:n]:
        print(f"{'─'*60}")
        print(f"prof_id:     {p['prof_id']}")
        print(f"Name:        {p['name']} — {p['institution']}")
        print(f"Method:      {p['method_type']}")
        print(f"Topic gap:   {p['topic_gap']}")
        print(f"Future work: {p['future_work']}")
        print(f"Vision:      {(p['research_vision'] or '')[:200]}...")
        print()



if __name__ == "__main__":
    import sys
    retry = "--retry" in sys.argv
    enrich(retry_failed=retry)
    inspect()
