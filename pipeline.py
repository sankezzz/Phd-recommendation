"""
pipeline.py — Student Profile → Professor Recommendations
Built with LangChain

Flow:
  1. Parse CV PDF          (LangChain Blob + PDFMinerParser)
  2. Split CV text         (LangChain RecursiveCharacterTextSplitter)
  3. Extract profile       (LangChain chain: prompt | ChatGroq | JsonOutputParser)
  4. Embed student         (HF SPECTER2)
  5. Find cluster          (cosine sim vs cluster_centroids.npy)
  6. Get candidates        (ChromaDB — filter by cluster, rank by vision similarity)
  7. Score complementarity (LangChain chain per candidate)
  8. Generate emails       (LangChain chain per top professor)
  9. Return top 10
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

# LangChain — LLM chains
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.2,
    max_tokens  = 500,              # sufficient for extraction + scoring
    api_key     = os.getenv("GROQ_API"),
)

hf_client = InferenceClient(
    model = "allenai/specter2_base",
    token = os.getenv("HF_TOKEN"),
)

chroma      = chromadb.PersistentClient(path="./chroma_db")
visions_col = chroma.get_collection("professor_visions")
CENTROIDS   = np.load("./vector_data/cluster_centroids.npy")   # shape (K, 768)

# ── Text splitter ─────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 3000,
    chunk_overlap = 100,
    separators    = ["\n\n", "\n", " ", ""]
)

# ── Step 1: Parse CV (PDF or TXT) ────────────────────────────────────────────

def parse_pdf(pdf_bytes: bytes) -> str:
    """LangChain native PDF parsing — no temp file needed."""
    blob   = Blob.from_data(pdf_bytes, mime_type="application/pdf")
    parser = PDFMinerParser()
    docs   = list(parser.lazy_parse(blob))
    return "\n".join(doc.page_content for doc in docs).strip()


def parse_txt(txt_bytes: bytes) -> str:
    """Decode plain text CV — try UTF-8 then fall back to latin-1."""
    try:
        return txt_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return txt_bytes.decode("latin-1").strip()


def parse_cv(file_bytes: bytes, file_name: str) -> str:
    """Route to correct parser based on file extension."""
    if file_name.lower().endswith(".txt"):
        return parse_txt(file_bytes)
    return parse_pdf(file_bytes)


# ── Step 2: Split CV text ─────────────────────────────────────────────────────

def get_cv_chunk(cv_text: str) -> str:
    """
    First chunk contains name, education, skills — most useful for extraction.
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
    topic_vector  ← broad_topics   → cluster matching
    vision_vector ← research_vision → reranking within cluster
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
    # Count professors in this specific cluster to avoid Chroma crash
    # when cluster has fewer members than n_candidates
    cluster_results = visions_col.get(where={"cluster_id": cluster_id})
    cluster_size    = len(cluster_results["ids"])
    safe_n          = max(1, min(n_candidates, cluster_size))

    results = visions_col.query(
        query_embeddings = [vision_vector.tolist()],
        n_results        = safe_n,
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


# ── Step 8: Generate full emails ──────────────────────────────────────────────

email_prompt = ChatPromptTemplate.from_messages([
    ("system", """You write cold emails from PhD applicants to professors.
Emails must sound human, specific, and concise — under 200 words.
Never use phrases like 'I am deeply passionate' or 'your groundbreaking work'.
No emojis. No subject line."""),

    ("human", """Write a cold email from this student to this professor.

STUDENT:
Name: {student_name}
Background: {background}
Research vision: {research_vision}

PROFESSOR: {prof_name} — {institution}
Opening hook: {email_hook}

The email must:
- Open with the hook referencing the professor's specific future work
- Explain in 2 sentences what the student brings
- End with a specific ask — 20 minute call or lab visit
- Sound human not AI generated
- Be under 200 words""")
])

# Higher temperature for email — more natural, less robotic
email_chain = email_prompt | llm.bind(temperature=0.6, max_tokens=400) | StrOutputParser()


def generate_emails(professors: list[dict], student: dict) -> list[dict]:
    """Generate full emails for top 2 only — avoids rate limits."""
    for i, prof in enumerate(professors):
        if i >= 2:
            prof["email"] = ""
            continue
        if not prof.get("email_hook"):
            prof["email"] = ""
            continue
        try:
            prof["email"] = email_chain.invoke({
                "student_name":    student.get("name") or "I",
                "background":      student.get("background") or "",
                "research_vision": student.get("research_vision") or "",
                "prof_name":       prof.get("name") or "",
                "institution":     prof.get("institution") or "",
                "email_hook":      prof.get("email_hook") or "",
            })
        except Exception as e:
            print(f"[generate_emails] {prof.get('name')}: {e}")
            prof["email"] = ""
    return professors


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(pdf_bytes: bytes, interests: str, file_name: str = "cv.pdf") -> dict:
    """
    Full pipeline: PDF bytes + interests text → top 10 professors with emails.

    Returns:
    {
      "student":    { structured student profile },
      "cluster_id": int,
      "professors": [ top 10 with scores, reasons, email hooks, full emails ]
    }
    """
    print("\n── Pipeline start ──────────────────────────────────")

    print("[1/7] Parsing CV...")
    cv_text = parse_cv(pdf_bytes, file_name)
    if not cv_text:
        return {"error": "Could not extract text from PDF"}

    print("[2/7] Extracting student profile...")
    student = extract_student_profile(cv_text, interests)
    if not student:
        return {"error": "Could not extract student profile"}
    print(f"      Name:   {student.get('name')}")
    print(f"      Topics: {student.get('broad_topics')}")
    print(f"      Method: {student.get('method_type')}")

    print("[3/7] Embedding student...")
    topic_vec, vision_vec = embed_student(student)
    if topic_vec is None:
        return {"error": "Could not embed student topics"}
    if vision_vec is None:
        vision_vec = topic_vec   # fallback to topic vector

    print("[4/7] Finding cluster...")
    cluster_id = find_cluster(topic_vec)
    print(f"      Cluster: {cluster_id}")

    print("[5/7] Getting candidates...")
    candidates = get_candidates(cluster_id, vision_vec, n_candidates=20)
    print(f"      {len(candidates)} candidates found")

    if not candidates:
        return {"error": f"No professors found in cluster {cluster_id}"}

    print("[6/7] Scoring complementarity...")
    professors = score_complementarity(candidates, student, top_n=10)
    print(f"      Top: {professors[0]['name'] if professors else 'none'}")

    print("[7/7] Generating emails...")
    professors = generate_emails(professors, student)

    print("── Done ────────────────────────────────────────────\n")

    return {
        "student":    student,
        "cluster_id": cluster_id,
        "professors": professors,
    }