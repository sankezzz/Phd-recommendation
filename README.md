## Deployed - https://sankezzzz-phd-recommend.hf.space/

# Setup Guide — PhD Professor Recommender

Match PhD applicants to professors based on CV and research interests.
Uses LLM-powered complementarity scoring and drafts cold emails to the top matches.

---

## What you need before starting

| Requirement | Where to get it |
|---|---|
| Python 3.11+ | [python.org](https://python.org) |
| Git + Git LFS | [git-scm.com](https://git-scm.com) / [git-lfs.com](https://git-lfs.com) |
| Groq API key | [console.groq.com](https://console.groq.com) — free tier works |
| Hugging Face token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — read access is enough |

---

## Step 1 — Install Git LFS

Git LFS is required because the vector database and embeddings are binary files.

```bash
git lfs install
```

---

## Step 2 — Clone the repo

```bash
git clone https://github.com/sankezzzz/Phd-recommendation
cd Phd-recommendation
```

After cloning, pull the LFS files (chroma_db, vector_data):

```bash
git lfs pull
```

You should see `chroma_db/` and `vector_data/` populate with actual binary files.
If they are tiny text files (~130 bytes), `git lfs pull` did not run — do it again.

---

## Step 3 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

---

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 — Set up API keys

Create a `.env` file in the project root:

```
GROQ_API=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- `GROQ_API` — from [console.groq.com](https://console.groq.com) → API Keys
- `HF_TOKEN` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read)

Never commit this file. It is already in `.gitignore`.

---

## Step 6 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to use it

1. **Upload your CV** as a PDF or TXT file
2. **Describe your research interests** in your own words — the more specific the better
3. Click **Find Professors**
4. Wait ~30 seconds — the pipeline parses your CV, embeds it, finds the nearest professor cluster, scores complementarity, and drafts emails
5. You get up to 10 ranked professors with reasons and cold email drafts for the top 2

---

## How it works

```
CV (PDF/TXT)  +  Research interests text
        ↓
  LangChain extracts structured student profile
  (broad topics, skills, vision, method type)
        ↓
  HF Inference API embeds the profile (SPECTER2)
        ↓
  Cosine similarity → nearest professor cluster
        ↓
  ChromaDB retrieves top 20 professors from that cluster
        ↓
  LLM scores complementarity for each professor
  (what this student offers that professor needs)
        ↓
  Top 10 professors ranked + cold emails drafted for top 2
```

**Models used:**
- `llama-3.3-70b-versatile` via Groq (profile extraction, scoring, email generation)
- `allenai/specter2_base` via HF Inference API (embeddings)

**Database:** 647 professors from MIT, Stanford, CMU, Berkeley, Oxford, Princeton, Yale, ETH, Toronto, UW, Columbia — pre-embedded and stored in ChromaDB.

---

## Rebuilding the professor database (optional)

You only need this if you want to add new professors or universities.
The repo already ships with a pre-built database — skip this for normal use.

```bash
python data_scripts/data_stage1.py     # scrape professor pages
python data_scripts/stage2_data.py     # enrich with LLM
python data_scripts/stage3.py          # embed + cluster
python migrate.py                      # load into ChromaDB
```

---

## Troubleshooting

**`InternalError: file is not a database`**
The chroma_db files are LFS pointers, not real files. Run:
```bash
git lfs pull
```

**`GROQ_API not set` or API errors**
Check your `.env` file is in the project root and the key is correct.

**`Could not extract text from PDF`**
The PDF might be scanned/image-based. Use a text-based PDF or convert to TXT first.

**`No professors found in cluster X`**
Your research area may be outside the covered domains. Try rephrasing your interests
to be closer to CS / ML / biology / robotics / NLP.

**App loads but results are empty**
Check terminal logs for LLM errors — usually a rate limit on the free Groq tier.
Wait a minute and try again.

---

## Project structure

```
.
├── app.py                  # Streamlit frontend
├── pipeline.py             # Full matching pipeline
├── requirements.txt        # Pinned dependencies
├── Dockerfile              # For HF Spaces deployment
├── chroma_db/              # Pre-built ChromaDB (LFS)
├── vector_data/            # Cluster centroids + embeddings (LFS)
│   └── cluster_centroids.npy
├── data/
│   └── professors_clustered.json
└── data_scripts/           # Scripts to rebuild the professor DB
```
