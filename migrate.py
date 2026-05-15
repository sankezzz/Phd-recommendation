"""
Stage 4: Migrate to ChromaDB
Reads:
  topic_embeddings.npz      — {prof_ids, vectors}
  vision_embeddings.npz     — {prof_ids, vectors}
  professors_clustered.json — metadata

Produces two Chroma collections:
  professor_topics    — topic vectors + basic metadata
  professor_visions   — vision vectors + full metadata
"""

import json
import numpy as np
import chromadb
from dotenv import load_dotenv

load_dotenv()

TOPIC_FILE  = "vector_data/topic_embeddings.npz"
VISION_FILE = "vector_data/vision_embeddings.npz"
META_FILE   = "data/professors_clustered.json"
CHROMA_DIR  = "./chroma_db"
BATCH_SIZE  = 50

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    # Load metadata
    with open(META_FILE) as f:
        professors = json.load(f)
    meta_index = {p["prof_id"]: p for p in professors}

    # Load topic embeddings
    t = np.load(TOPIC_FILE)
    topic_ids     = t["prof_ids"].tolist()
    topic_vectors = t["vectors"].tolist()

    # Load vision embeddings
    v = np.load(VISION_FILE)
    vision_ids     = v["prof_ids"].tolist()
    vision_vectors = v["vectors"].tolist()

    print(f"Loaded {len(professors)} professor records")
    print(f"topic_embeddings.npz  — {len(topic_ids)} vectors")
    print(f"vision_embeddings.npz — {len(vision_ids)} vectors")

    return meta_index, topic_ids, topic_vectors, vision_ids, vision_vectors


# ── Batch upsert helper ───────────────────────────────────────────────────────

def upsert_batch(collection, ids, embeddings, metadatas):
    for i in range(0, len(ids), BATCH_SIZE):
        collection.upsert(
            ids        = ids[i:i+BATCH_SIZE],
            embeddings = embeddings[i:i+BATCH_SIZE],
            metadatas  = metadatas[i:i+BATCH_SIZE],
        )
    print(f"  → {len(ids)} records upserted into {collection.name}")


# ── Migrate ───────────────────────────────────────────────────────────────────

def migrate(reset: bool = False):
    meta_index, topic_ids, topic_vectors, vision_ids, vision_vectors = load_data()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        print("\nResetting collections...")
        for name in ["professor_topics", "professor_visions"]:
            try:
                client.delete_collection(name)
                print(f"  Deleted {name}")
            except:
                pass

    topics_col  = client.get_or_create_collection(
        name     = "professor_topics",
        metadata = {"hnsw:space": "cosine"}
    )
    visions_col = client.get_or_create_collection(
        name     = "professor_visions",
        metadata = {"hnsw:space": "cosine"}
    )

    # ── Collection 1: professor_topics ────────────────────────────────────────
    print("\nLoading professor_topics...")
    t_ids, t_vecs, t_metas = [], [], []

    for prof_id, vector in zip(topic_ids, topic_vectors):
        prof = meta_index.get(prof_id, {})
        t_ids.append(str(prof_id))
        t_vecs.append(vector)
        t_metas.append({
            "prof_id":     prof_id,
            "name":        prof.get("name") or "",
            "institution": prof.get("institution") or "",
            "cluster_id":  prof.get("cluster_id") if prof.get("cluster_id") is not None else -1,
        })

    upsert_batch(topics_col, t_ids, t_vecs, t_metas)

    # ── Collection 2: professor_visions ───────────────────────────────────────
    print("\nLoading professor_visions...")
    v_ids, v_vecs, v_metas = [], [], []

    for prof_id, vector in zip(vision_ids, vision_vectors):
        prof = meta_index.get(prof_id, {})

        # Chroma metadata must be scalar — convert lists to pipe-separated strings
        future_work    = " | ".join(prof.get("future_work") or [])
        openalex_topics= " | ".join(prof.get("openalex_topics") or [])
        topic_gap      = " | ".join(prof.get("topic_gap") or [])

        v_ids.append(str(prof_id))
        v_vecs.append(vector)
        v_metas.append({
            "prof_id":         prof_id,
            "name":            prof.get("name") or "",
            "institution":     prof.get("institution") or "",
            "cluster_id":      prof.get("cluster_id") if prof.get("cluster_id") is not None else -1,
            "method_type":     prof.get("method_type") or "",
            "future_work":     future_work,
            "openalex_topics": openalex_topics,
            "topic_gap":       topic_gap,
        })

    upsert_batch(visions_col, v_ids, v_vecs, v_metas)

    print(f"\nDone. Chroma DB at: {CHROMA_DIR}")


# ── Verify ────────────────────────────────────────────────────────────────────

def verify():
    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    topics_col  = client.get_collection("professor_topics")
    visions_col = client.get_collection("professor_visions")

    print(f"\nChroma DB:")
    print(f"  professor_topics  — {topics_col.count()} records")
    print(f"  professor_visions — {visions_col.count()} records")

    # Sample query
    sample = topics_col.get(limit=1, include=["embeddings", "metadatas"])
    if sample["ids"]:
        name   = sample["metadatas"][0]["name"]
        emb    = sample["embeddings"][0]
        print(f"\nSample — nearest professors to {name}:")
        results = topics_col.query(query_embeddings=[emb], n_results=4)
        for rid, meta in zip(results["ids"][0], results["metadatas"][0]):
            print(f"  [{rid}] {meta['name']} — {meta['institution']} — cluster {meta['cluster_id']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    reset = "--reset" in sys.argv
    migrate(reset=reset)
    verify()