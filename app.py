"""
app.py — Streamlit frontend
Minimal, clean, no emojis.
"""

import streamlit as st
from pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "PhD Match",
    page_icon  = None,
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f0f0f;
    color: #e0e0e0;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 3rem; max-width: 860px; }

  /* Title */
  .title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
  }
  .subtitle {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 3rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Section labels */
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  /* Input areas */
  .stTextArea textarea, .stFileUploader {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 2px !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
  }

  /* Button */
  .stButton > button {
    background-color: #ffffff !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    margin-top: 1rem !important;
  }
  .stButton > button:hover {
    background-color: #d0d0d0 !important;
  }

  /* Divider */
  hr {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
  }

  /* Student profile card */
  .profile-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
  }
  .profile-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
  }
  .profile-card .value {
    font-size: 0.88rem;
    color: #c0c0c0;
    margin-bottom: 0.9rem;
  }
  .profile-card .tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #888;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem;
  }

  /* Professor card */
  .prof-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
  }
  .prof-card:hover {
    border-color: #2a2a2a;
  }
  .prof-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #333;
    position: absolute;
    top: 1.4rem;
    right: 1.6rem;
  }
  .prof-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: #e8e8e8;
    margin-bottom: 0.1rem;
  }
  .prof-institution {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555;
    margin-bottom: 0.9rem;
  }
  .prof-score {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    background: #1a1a1a;
    border: 1px solid #252525;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.9rem;
  }
  .prof-reason {
    font-size: 0.84rem;
    color: #888;
    line-height: 1.55;
    margin-bottom: 0.9rem;
  }
  .prof-email-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #333;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }
  .prof-email {
    font-size: 0.84rem;
    color: #a0a0a0;
    line-height: 1.6;
    white-space: pre-wrap;
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Error */
  .error-msg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #c0392b;
    background: #1a0e0e;
    border: 1px solid #2a1515;
    border-radius: 2px;
    padding: 0.8rem 1rem;
  }

  /* Progress */
  .stSpinner > div { border-top-color: #444 !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="title">PhD Match</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your CV and describe your research interests to find matching professors.</div>', unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">CV — PDF only</div>', unsafe_allow_html=True)
pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Research interests — in your own words</div>', unsafe_allow_html=True)
interests = st.text_area(
    "",
    placeholder="Describe what you want to work on, what problems interest you, what methods you want to use...",
    height=130,
    label_visibility="collapsed"
)

run = st.button("Find Professors")


# ── Run pipeline ──────────────────────────────────────────────────────────────

if run:
    if not pdf_file:
        st.markdown('<div class="error-msg">Please upload your CV as a PDF.</div>', unsafe_allow_html=True)
    elif not interests.strip():
        st.markdown('<div class="error-msg">Please describe your research interests.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing your profile and finding matches..."):
            result = run_pipeline(pdf_file.read(), interests)

        if "error" in result:
            st.markdown(f'<div class="error-msg">{result["error"]}</div>', unsafe_allow_html=True)
        else:
            student    = result["student"]
            professors = result["professors"]

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Student profile summary ───────────────────────────────────
            st.markdown('<div class="section-label">Your profile</div>', unsafe_allow_html=True)

            topics_html = "".join(
                f'<span class="tag">{t}</span>'
                for t in (student.get("broad_topics") or [])
            )
            skills_html = "".join(
                f'<span class="tag">{s}</span>'
                for s in (student.get("skills") or [])[:8]
            )

            st.markdown(f"""
            <div class="profile-card">
              <div class="label">Research areas</div>
              <div class="value">{topics_html}</div>
              <div class="label">Skills</div>
              <div class="value">{skills_html}</div>
              <div class="label">Research vision</div>
              <div class="value">{student.get("research_vision") or ""}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Professor results ─────────────────────────────────────────
            st.markdown(
                f'<div class="section-label">{len(professors)} professors matched</div>',
                unsafe_allow_html=True
            )

            for i, prof in enumerate(professors, 1):
                score    = prof.get("complementarity_score", 0)
                sim      = prof.get("vision_similarity", 0)
                email    = prof.get("email", "") or prof.get("email_hook", "")

                # future work tags
                fw_html = "".join(
                    f'<span class="tag">{fw}</span>'
                    for fw in (prof.get("future_work") or [])[:3]
                )

                st.markdown(f"""
                <div class="prof-card">
                  <div class="prof-rank">#{i}</div>
                  <div class="prof-name">{prof.get("name")}</div>
                  <div class="prof-institution">{prof.get("institution")}</div>
                  <div class="prof-score">fit {score}/10 &nbsp;&nbsp; similarity {sim}</div>
                  <div class="prof-reason">{prof.get("reason") or ""}</div>
                  {f'<div style="margin-bottom:0.6rem">{fw_html}</div>' if fw_html else ""}
                  <div class="prof-email-label">Email opening</div>
                  <div class="prof-email">{email}</div>
                </div>
                """, unsafe_allow_html=True)