"""
Stage 1: Professor Data Collection — OpenAlex only
Run locally: python data.py
Saves after every professor — safe to Ctrl+C and resume.
"""

import requests
import time
import json
import os

OPENALEX_BASE   = "https://api.openalex.org"
OUTPUT_FILE     = "professors_raw.json"
PAPERS_PER_PROF = 5
MIN_WORKS       = 15

TARGET_INSTITUTIONS = {
    "MIT":       "I63966007",
    "Stanford":  "I97018004",
    "CMU":       "I74973139",
    "Berkeley":  "I95457486",
    "Princeton": "I20089843",
    "Toronto":   "I185261750",
    "Oxford":    "I33213144",
    "ETH":       "I129801699",
    "Yale":       "I32971472",
   " Columbia University": "I78577930",
"University of Washington": "I201448701"

}

# ── OpenAlex ──────────────────────────────────────────────────────────────────

def oa_get(endpoint, params):
    params["mailto"] = "your@email.com"
    r = requests.get(f"{OPENALEX_BASE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def decode_abstract(inv_index: dict) -> str:
    if not inv_index:
        return ""
    positions = {}
    for word, locs in inv_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_authors(inst_id: str, inst_name: str, limit: int = 60) -> list:
    print(f"\n[OpenAlex] Fetching {inst_name}...")
    authors, cursor = [], "*"

    while len(authors) < limit:
        data = oa_get("authors", {
            "filter":   f"last_known_institutions.id:{inst_id},works_count:>{MIN_WORKS}",
            "sort":     "cited_by_count:desc",
            "per_page": 50,
            "cursor":   cursor,
            "select":   "id,display_name,works_count,cited_by_count,topics",
        })
        results = data.get("results", [])
        if not results:
            break
        authors += [a for a in results if a.get("topics")]
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    print(f"  → {min(len(authors), limit)} authors found")
    return authors[:limit]


def fetch_papers(author_id: str) -> list:
    data = oa_get("works", {
        "filter":   f"authorships.author.id:{author_id}",
        "sort":     "publication_year:desc",
        "per_page": PAPERS_PER_PROF,
        "select":   "id,title,publication_year,abstract_inverted_index,primary_location,topics",
    })
    papers = []
    for w in data.get("results", []):
        abstract = decode_abstract(w.get("abstract_inverted_index") or {})
        venue    = (w.get("primary_location") or {}).get("source") or {}
        papers.append({
            "openalex_id": w.get("id"),
            "title":       w.get("title"),
            "year":        w.get("publication_year"),
            "abstract":    abstract,
            "venue_name":  venue.get("display_name"),
            "venue_type":  venue.get("type"),         # "journal" | "conference" | "repository"
            "topics":      [t["display_name"] for t in w.get("topics", [])[:3]],
        })
    return papers


# ── Save ──────────────────────────────────────────────────────────────────────

def save(data: list):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUTPUT_FILE)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_professor_db(institutions=TARGET_INSTITUTIONS, max_per_inst=60):

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_profs = json.load(f)
        seen = {p["openalex_id"] for p in all_profs}
        print(f"Resuming — {len(all_profs)} already saved")
    else:
        all_profs, seen = [], set()

    for inst_name, inst_id in institutions.items():
        authors = fetch_authors(inst_id, inst_name, max_per_inst)

        for i, author in enumerate(authors):
            oa_id = author["id"]

            if oa_id in seen:
                print(f"  [{i+1}/{len(authors)}] {author['display_name']} — skip")
                continue

            name = author["display_name"]
            papers = fetch_papers(oa_id)

            # Filter out papers with no abstract — not useful for Stage 2
            papers_with_abstract = [p for p in papers if p["abstract"].strip()]

            professor = {
                "openalex_id":     oa_id,
                "name":            name,
                "institution":     inst_name,
                "institution_id":  inst_id,
                "openalex_topics": [t["display_name"] for t in author.get("topics", [])[:5]],
                "works_count":     author.get("works_count"),
                "cited_by_count":  author.get("cited_by_count"),
                "papers":          papers_with_abstract,
                # Stage 2 — filled by enrich.py
                "research_vision": None,
                "method_type":     None,
                "future_work":     None,
                "embedding":       None,
            }

            all_profs.append(professor)
            seen.add(oa_id)
            save(all_profs)

            abstract_count = len(papers_with_abstract)
            print(f"  [{i+1}/{len(authors)}] {name} — {abstract_count} papers w/ abstracts")

    print(f"\nDone. {len(all_profs)} professors saved to {OUTPUT_FILE}")
    return all_profs


def inspect(n=3):
    with open(OUTPUT_FILE) as f:
        profs = json.load(f)

    print(f"\nTotal saved: {len(profs)}\n")
    for p in profs[:n]:
        print(f"{'─'*60}")
        print(f"{p['name']} — {p['institution']}")
        print(f"Topics:  {p['openalex_topics']}")
        print(f"Papers with abstracts: {len(p['papers'])}")
        if p["papers"]:
            latest = p["papers"][0]
            print(f"Latest:  {latest['title']} ({latest['year']})")
            print(f"Venue:   {latest['venue_name']} ({latest['venue_type']})")
            print(f"Abstract preview: {latest['abstract'][:200]}...")
        print()


if __name__ == "__main__":
    build_professor_db()
    inspect()