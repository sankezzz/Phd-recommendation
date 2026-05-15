"""
Stage 3: Generate Embeddings + Cluster
Reads professors_enriched.json

Produces:
  topic_embeddings.npz     — {prof_ids, vectors}  shape (N, 768)
  vision_embeddings.npz    — {prof_ids, vectors}  shape (N, 768)
  professors_clustered.json — metadata only, NO embeddings, has cluster_id
  cluster_centroids.npy    — shape (K, 768), used to map student to cluster
"""

import json
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN    = os.getenv("HF_TOKEN")
HF_MODEL    = "allenai/specter2_base"
INPUT_FILE  = "professors_enriched.json"
META_FILE   = "professors_clustered.json"   # metadata only, no embeddings
TOPIC_FILE  = "topic_embeddings.npz"
VISION_FILE = "vision_embeddings.npz"
CENTROID_FILE = "cluster_centroids.npy"
N_CLUSTERS  = 40
HF_DELAY    = 1.5

client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

# ── Embed ─────────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float] | None:
    if not text or not text.strip():
        return None
    try:
        result = client.feature_extraction(text)
        vector = np.array(result)
        if vector.ndim > 1:
            vector = vector.mean(axis=0)
        return vector.tolist()
    except Exception as e:
        print(f"\n    [Embed error] {e}")
        return None

# ── Save helpers ──────────────────────────────────────────────────────────────

def save_json(data: list, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_npz(prof_ids: list, vectors: list, path: str):
    np.savez(
        path,
        prof_ids = np.array(prof_ids, dtype=np.int32),
        vectors  = np.array(vectors,  dtype=np.float32)
    )
    print(f"  Saved {path} — {len(prof_ids)} professors, {len(vectors[0])}d vectors")


# ── Step 1 + 2: Generate embeddings ──────────────────────────────────────────

def generate_embeddings():
    with open(INPUT_FILE) as f:
        professors = json.load(f)
    print(f"Loaded {len(professors)} professors\n")

    # Load existing progress if resuming
    # Track which prof_ids already have embeddings
    existing_topic_ids  = set()
    existing_vision_ids = set()

    topic_prof_ids,  topic_vectors  = [], []
    vision_prof_ids, vision_vectors = [], []

    if os.path.exists(TOPIC_FILE):
        t = np.load(TOPIC_FILE)
        topic_prof_ids  = t["prof_ids"].tolist()
        topic_vectors   = t["vectors"].tolist()
        existing_topic_ids = set(topic_prof_ids)
        print(f"Resuming topic  — {len(existing_topic_ids)} already embedded")

    if os.path.exists(VISION_FILE):
        v = np.load(VISION_FILE)
        vision_prof_ids  = v["prof_ids"].tolist()
        vision_vectors   = v["vectors"].tolist()
        existing_vision_ids = set(vision_prof_ids)
        print(f"Resuming vision — {len(existing_vision_ids)} already embedded")

    for prof in professors:
        prof_id   = prof["prof_id"]
        name      = prof["name"]
        need_topic  = prof_id not in existing_topic_ids
        need_vision = prof_id not in existing_vision_ids

        if not need_topic and not need_vision:
            print(f"  skip [{prof_id}] {name}")
            continue

        print(f"  [{prof_id}] {name}", end="", flush=True)

        # ── Topic embedding ────────────────────────────────────────────────
        if need_topic:
            topics_text = ", ".join(prof.get("openalex_topics") or [])
            topic_emb   = None
            if topics_text:
                time.sleep(HF_DELAY)
                topic_emb = embed(topics_text)
            if topic_emb:
                topic_prof_ids.append(prof_id)
                topic_vectors.append(topic_emb)
                existing_topic_ids.add(prof_id)
                print(f" — topic ✓", end="", flush=True)
            else:
                print(f" — topic ✗", end="", flush=True)

        # ── Vision embedding ───────────────────────────────────────────────
        if need_vision:
            vision_text = prof.get("research_vision") or ""
            vision_emb  = None
            if vision_text:
                time.sleep(HF_DELAY)
                vision_emb = embed(vision_text)
            if vision_emb:
                vision_prof_ids.append(prof_id)
                vision_vectors.append(vision_emb)
                existing_vision_ids.add(prof_id)
                print(f" — vision ✓", end="", flush=True)
            else:
                print(f" — vision ✗", end="", flush=True)

        print()

        # Save after every professor — safe to interrupt
        if topic_vectors:
            save_npz(topic_prof_ids, topic_vectors, TOPIC_FILE)
        if vision_vectors:
            save_npz(vision_prof_ids, vision_vectors, VISION_FILE)

    print(f"\nEmbedding complete:")
    print(f"  topic_embeddings.npz  — {len(topic_prof_ids)} professors")
    print(f"  vision_embeddings.npz — {len(vision_prof_ids)} professors")

    return topic_prof_ids, topic_vectors, vision_prof_ids, vision_vectors


# ── Step 3: K-Means on topic embeddings ──────────────────────────────────────

def run_clustering():
    # Load topic embeddings
    t        = np.load(TOPIC_FILE)
    prof_ids = t["prof_ids"].tolist()
    vectors  = t["vectors"]

    print(f"\nClustering {len(prof_ids)} professors with K={N_CLUSTERS}...")

    vectors_norm = normalize(vectors)
    k            = min(N_CLUSTERS, len(prof_ids))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors_norm)

    # Save centroids
    centroids = normalize(kmeans.cluster_centers_).astype(np.float32)
    np.save(CENTROID_FILE, centroids)
    print(f"Saved {k} centroids → {CENTROID_FILE}")

    # Build cluster_id lookup
    cluster_map = {pid: int(label) for pid, label in zip(prof_ids, labels)}

    # Load professors metadata, strip embeddings, add cluster_id
    with open(INPUT_FILE) as f:
        professors = json.load(f)

    meta = []
    for prof in professors:
        pid = prof["prof_id"]
        record = {k: v for k, v in prof.items()
                  if k not in ("embedding", "topic_embedding", "vision_embedding")}
        record["cluster_id"] = cluster_map.get(pid)
        meta.append(record)

    save_json(meta, META_FILE)
    print(f"Saved metadata → {META_FILE} (no embeddings)")

    # Cluster summary
    cluster_members = {}
    for pid, label in zip(prof_ids, labels):
        cluster_members.setdefault(int(label), []).append(pid)

    prof_index = {p["prof_id"]: p for p in meta}
    print(f"\nCluster summary:")
    for cid in sorted(cluster_members):
        members      = cluster_members[cid]
        sample       = prof_index.get(members[0], {})
        topics       = sample.get("openalex_topics", [])[:3]
        print(f"  Cluster {cid:2d} — {len(members):3d} professors | {topics}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--cluster-only" in sys.argv:
        run_clustering()
    else:
        generate_embeddings()
        run_clustering()