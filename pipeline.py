"""
pipeline.py — Student Profile → Professor Recommendations
Built with LangChain

Flow:
  1. Parse CV PDF        (LangChain Blob + PDFMinerParser)
  2. Split CV text       (LangChain RecursiveCharacterTextSplitter)
  3. Extract profile     (LangChain chain: prompt | ChatGroq | JsonOutputParser)
  4. Embed student       (HF SPECTER2)
  5. Find cluster        (cosine sim vs cluster_centroids.npy)
  6. Get candidates      (ChromaDB — filter by cluster, rank by vision similarity)
  7. Score complementarity (LangChain chain per candidate)
  8. Return top 10
"""

import os
import numpy as np
import chromadb
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from huggingface_hub import InferenceClient

# LangChain — PDF parsing
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import PDFMinerParser

# LangChain — text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain — LLM chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.2,
    max_tokens  = 2000,
    api_key     = os.getenv("GROQ_API"),
)

hf_client = InferenceClient(
    model = "allenai/specter2_base",
    token = os.getenv("HF_TOKEN"),
)

chroma      = chromadb.PersistentClient(path="./chroma_db")
visions_col = chroma.get_collection("professor_visions")
CENTROIDS   = np.load("cluster_centroids.npy")   # shape (K, 768)

# ── Text splitter ─────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 3000,
    chunk_overlap = 100,
    separators    = ["\n\n", "\n", " ", ""]
)

# ── Step 1: Parse CV PDF ──────────────────────────────────────────────────────

def parse_pdf(pdf_bytes: bytes) -> str:
    """
    LangChain native PDF parsing — no temp file needed.
    Blob.from_data() takes raw bytes directly from Streamlit uploader.
    """
    blob   = Blob.from_data(pdf_bytes, mime_type="application/pdf")
    parser = PDFMinerParser()
    docs   = list(parser.lazy_parse(blob))
    return "\n".join(doc.page_content for doc in docs).strip()


# ── Step 2: Split + trim CV text ──────────────────────────────────────────────

def get_cv_chunk(cv_text: str) -> str:
    """
    Split CV into chunks, return first chunk.
    First chunk always contains name, education, skills — most useful for extraction.
    Cleaner than raw slicing cv_text[:3000].
    """
    chunks = splitter.split_text(cv_text)
    return chunks[0] if chunks else cv_text


# ── Step 3: Extract student profile ──────────────────────────────────────────

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research profile analyst.
Extract structured information from a student's CV and research interests.
Always respond with valid JSON only, no explanation, no markdown."""),

    ("human", """CV TEXT:
{cv_chunk}

STUDENT'S OWN DESCRIPTION OF INTERESTS:
{interests}

Extract the following JSON:
{{
  "name":            "student name if found, else null",
  "broad_topics":    ["3-5 broad research areas e.g. Machine Learning, NLP, Biology"],
  "skills":          ["specific technical skills: tools, languages, methods"],
  "background":      "1-2 sentences on academic background and domain",
  "research_vision": "100 words max: what research problems they want to work on and why",
  "method_type":     "one of: theoretical / empirical / systems / interdisciplinary",
  "what_they_offer": ["3-5 specific things this student brings: unique skills, domain knowledge, prior work"]
}}

Rules:
- broad_topics must be academic research fields not job skills
- what_they_offer must be specific not generic
- research_vision must reflect what they WANT to do, not what they have done
- Return ONLY the JSON object""")
])

extract_chain = extract_prompt | llm | JsonOutputParser()


def extract_student_profile(cv_text: str, interests: str) -> dict | None:
    """Run extraction chain — CV chunk + interests → structured student profile."""
    try:
        cv_chunk = get_cv_chunk(cv_text)
        return extract_chain.invoke({
            "cv_chunk":  cv_chunk,
            "interests": interests,
        })
    except Exception as e:
        print(f"[extract_student_profile] Error: {e}")
        return None


# ── Step 4: Embed student ─────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray | None:
    if not text or not text.strip():
        return None
    try:
        result = hf_client.feature_extraction(text)
        vector = np.array(result)
        if vector.ndim > 1:
            vector = vector.mean(axis=0)
        return vector.astype(np.float32)
    except Exception as e:
        print(f"[embed] Error: {e}")
        return None


def embed_student(profile: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    topic_vector  ← broad_topics joined     → cluster matching
    vision_vector ← research_vision text    → reranking within cluster
    """
    topic_vec  = embed(", ".join(profile.get("broad_topics") or []))
    vision_vec = embed(profile.get("research_vision") or "")
    return topic_vec, vision_vec


# ── Step 5: Find nearest cluster ─────────────────────────────────────────────

def find_cluster(topic_vector: np.ndarray) -> int:
    student_norm = normalize(topic_vector.reshape(1, -1))
    similarities = CENTROIDS @ student_norm.T
    return int(np.argmax(similarities))


# ── Step 6: Get candidates from cluster ──────────────────────────────────────

def get_candidates(cluster_id: int, vision_vector: np.ndarray,
                   n_candidates: int = 20) -> list[dict]:
    results = visions_col.query(
        query_embeddings = [vision_vector.tolist()],
        n_results        = n_candidates,
        where            = {"cluster_id": cluster_id},
        include          = ["metadatas", "distances"],
    )

    candidates = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        candidates.append({
            **meta,
            "vision_similarity": round(1 - dist, 4),
            "future_work":     [x.strip() for x in meta.get("future_work", "").split("|") if x.strip()],
            "openalex_topics": [x.strip() for x in meta.get("openalex_topics", "").split("|") if x.strip()],
            "topic_gap":       [x.strip() for x in meta.get("topic_gap", "").split("|") if x.strip()],
        })

    return candidates


# ── Step 7: Complementarity scoring ──────────────────────────────────────────

comp_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research fit analyst.
Score how well a student fills a professor's specific research needs.
Respond with valid JSON only, no explanation, no markdown."""),

    ("human", """STUDENT:
Background: {background}
Skills: {skills}
What they offer: {what_they_offer}
Research vision: {research_vision}

PROFESSOR: {prof_name} — {institution}
Research topics: {prof_topics}
Future work / open problems: {future_work}
New directions: {topic_gap}

Return ONLY this JSON:
{{
  "complementarity_score": <int 1-10>,
  "reason": "1-2 sentences: what specific thing this student brings that this professor needs",
  "email_hook": "1 sentence opening a cold email — references professor's specific future work and student's specific skill"
}}

- complementarity_score reflects what student OFFERS that professor NEEDS, not just topic overlap
- reason and email_hook must be specific, never generic""")
])

comp_chain = comp_prompt | llm | JsonOutputParser()


def score_complementarity(candidates: list[dict], student: dict,
                           top_n: int = 10) -> list[dict]:
    scored = []

    for prof in candidates:
        future_work = prof.get("future_work") or []
        topic_gap   = prof.get("topic_gap") or []

        if not future_work and not topic_gap:
            prof.update({"complementarity_score": 0, "reason": "", "email_hook": ""})
            scored.append(prof)
            continue

        try:
            result = comp_chain.invoke({
                "background":      student.get("background") or "",
                "skills":          ", ".join(student.get("skills") or []),
                "what_they_offer": ", ".join(student.get("what_they_offer") or []),
                "research_vision": student.get("research_vision") or "",
                "prof_name":       prof.get("name") or "",
                "institution":     prof.get("institution") or "",
                "prof_topics":     ", ".join(prof.get("openalex_topics") or []),
                "future_work":     " | ".join(future_work),
                "topic_gap":       " | ".join(topic_gap),
            })
            prof.update({
                "complementarity_score": result.get("complementarity_score", 0),
                "reason":     result.get("reason", ""),
                "email_hook": result.get("email_hook", ""),
            })
        except Exception as e:
            print(f"[score_complementarity] {prof.get('name')}: {e}")
            prof.update({"complementarity_score": 0, "reason": "", "email_hook": ""})

        scored.append(prof)

    scored.sort(key=lambda x: x.get("complementarity_score", 0), reverse=True)
    return scored[:top_n]


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(pdf_bytes: bytes, interests: str) -> dict:
    """
    Full pipeline: PDF bytes + interests text → top 10 professor recommendations.

    Returns:
    {
      "student":    { structured student profile },
      "cluster_id": int,
      "professors": [ top 10 with scores, reasons, email hooks ]
    }
    """
    print("\n── Pipeline start ──────────────────────────────────")

    print("[1/6] Parsing CV...")
    cv_text = parse_pdf(pdf_bytes)
    if not cv_text:
        return {"error": "Could not extract text from PDF"}

    print("[2/6] Extracting student profile...")
    student = extract_student_profile(cv_text, interests)
    if not student:
        return {"error": "Could not extract student profile"}
    print(f"      Name:   {student.get('name')}")
    print(f"      Topics: {student.get('broad_topics')}")
    print(f"      Method: {student.get('method_type')}")

    print("[3/6] Embedding student...")
    topic_vec, vision_vec = embed_student(student)
    if topic_vec is None:
        return {"error": "Could not embed student topics"}
    if vision_vec is None:
        vision_vec = topic_vec

    print("[4/6] Finding cluster...")
    cluster_id = find_cluster(topic_vec)
    print(f"      Cluster: {cluster_id}")

    print("[5/6] Getting candidates...")
    candidates = get_candidates(cluster_id, vision_vec, n_candidates=20)
    print(f"      {len(candidates)} candidates found")

    print("[6/6] Scoring complementarity...")
    professors = score_complementarity(candidates, student, top_n=10)
    print(f"      Top: {professors[0]['name'] if professors else 'none'}")
    print("── Done ────────────────────────────────────────────\n")

    return {
        "student":    student,
        "cluster_id": cluster_id,
        "professors": professors,
    }