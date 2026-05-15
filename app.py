"""
app.py — PhD Match Frontend
"""

import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="PhD Match", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f0f0f;
    color: #e8e8e8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 3rem; max-width: 900px; }

.stTextArea textarea {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea::placeholder { color: #555 !important; }
.stTextArea textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}
.stButton > button {
    background-color: #ffffff !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2.5rem !important;
    margin-top: 1.2rem !important;
}
.stButton > button:hover { background-color: #ddd !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def lbl(text):
    st.markdown(
        f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
        f'color:#666;text-transform:uppercase;letter-spacing:0.12em;'
        f'margin:1.2rem 0 0.4rem 0">{text}</p>',
        unsafe_allow_html=True
    )

def tags_html(items, max_items=8):
    style = (
        "display:inline-block;background:#1e1e1e;border:1px solid #2a2a2a;"
        "border-radius:2px;font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
        "color:#aaa;padding:0.2rem 0.55rem;margin:0.15rem"
    )
    html = '<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:0.5rem">'
    for item in (items or [])[:max_items]:
        html += f'<span style="{style}">{item}</span>'
    html += '</div>'
    return html

def divider():
    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e1e1e;margin:2rem 0">',
        unsafe_allow_html=True
    )

def section_label(text):
    return (
        f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;'
        f'color:#444;text-transform:uppercase;letter-spacing:0.12em;'
        f'margin:1rem 0 0.35rem 0">{text}</p>'
    )

def body_text(text, muted=False):
    color = "#888" if muted else "#bbb"
    return (
        f'<p style="font-size:0.88rem;color:{color};'
        f'line-height:1.65;margin:0 0 0.5rem 0">{text}</p>'
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;'
    'font-weight:500;color:#ffffff;letter-spacing:0.08em;'
    'text-transform:uppercase;margin-bottom:0.3rem">Find the Prof That wants you</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
    'color:#777;margin-bottom:3rem">Upload your CV and describe your research '
    'interests to find matching professors.</p>',
    unsafe_allow_html=True
)

# ── Inputs ────────────────────────────────────────────────────────────────────

lbl("CV — PDF or TXT")
cv_file = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")

lbl("Research interests — in your own words")
interests = st.text_area(
    "",
    placeholder="Describe what you want to work on, what problems interest you, what methods you want to use...",
    height=140,
    label_visibility="collapsed"
)

run = st.button("Find Professors")

# ── Pipeline ──────────────────────────────────────────────────────────────────

if run:
    if not cv_file:
        st.error("Please upload your CV as a PDF or TXT file.")
    elif not interests.strip():
        st.error("Please describe your research interests.")
    else:
        with st.spinner("Analyzing your profile and finding matches..."):
            result = run_pipeline(cv_file.read(), interests, cv_file.name)

        if "error" in result:
            st.error(result["error"])
        else:
            student    = result["student"]
            professors = result["professors"]
            cluster_id = result["cluster_id"]

            divider()

            # ── Student profile ───────────────────────────────────────────
            st.markdown(
                '<p style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
                'color:#666;text-transform:uppercase;letter-spacing:0.12em;'
                'margin-bottom:0.8rem">Your profile</p>',
                unsafe_allow_html=True
            )

            profile_html = (
                '<div style="background:#131313;border:1px solid #222;border-radius:2px;padding:1.5rem 1.8rem">'
                + section_label("Research areas")
                + tags_html(student.get("broad_topics"))
                + section_label("Skills")
                + tags_html(student.get("skills"), max_items=8)
                + section_label("Research vision")
                + body_text(student.get("research_vision") or "")
                + '</div>'
            )
            st.markdown(profile_html, unsafe_allow_html=True)

            divider()

            # ── Results header ────────────────────────────────────────────
            st.markdown(
                '<h1 style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;'
                'font-weight:500;color:#ffffff;letter-spacing:0.08em;'
                'text-transform:uppercase;margin-bottom:0.3rem">Recommendations with email drafts for top 3</h1>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
                f'color:#666;text-transform:uppercase;letter-spacing:0.1em;'
                f'margin-bottom:1.2rem">'
                f'{len(professors)} professors matched &nbsp;/&nbsp; '
                f'cluster {cluster_id} &nbsp;/&nbsp; '
                f'emails drafted for top 2</p>',
                unsafe_allow_html=True
            )

            # ── Professor cards ───────────────────────────────────────────
            for i, prof in enumerate(professors, 1):
                score    = prof.get("complementarity_score", 0)
                sim      = prof.get("vision_similarity", 0)
                reason   = prof.get("reason", "")
                email    = prof.get("email", "")
                hook     = prof.get("email_hook", "")
                method   = prof.get("method_type", "")
                fw_list  = prof.get("future_work") or []
                gap_list = prof.get("topic_gap") or []

                fw_text   = " — ".join(fw_list[:3]) if fw_list else ""
                gap_tags  = tags_html(gap_list[:4]) if gap_list else ""

                if i == 1:
                    note = "Top match — highest complementarity score and vision alignment."
                elif i == 2:
                    note = "Second match — strong skill overlap with professor's stated future work."
                else:
                    note = f"Ranked {i} by complementarity. No email drafted."

                # Build card HTML piece by piece — avoids f-string nesting issues
                card = '<div style="background:#111;border:1px solid #1e1e1e;border-radius:2px;padding:1.6rem 1.8rem;margin-bottom:0.75rem;position:relative">'

                # Rank
                card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#2a2a2a;position:absolute;top:1.6rem;right:1.8rem">#{i}</p>'

                # Name + institution
                card += f'<p style="font-size:1rem;font-weight:500;color:#f0f0f0;margin:0 0 0.1rem 0">{prof.get("name","")}</p>'
                card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin:0 0 1rem 0">{prof.get("institution","")}</p>'

                # Badges
                badge_style = "font-family:IBM Plex Mono,monospace;font-size:0.65rem;background:#1a1a1a;border:1px solid #252525;border-radius:2px;padding:0.2rem 0.6rem;margin-right:0.4rem"
                card += f'<div style="margin-bottom:1rem">'
                card += f'<span style="{badge_style};color:#ccc">fit {score}/10</span>'
                card += f'<span style="{badge_style};color:#666">similarity {sim}</span>'
                if method:
                    card += f'<span style="{badge_style};color:#555">{method}</span>'
                card += '</div>'

                # Why this professor
                card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#444;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.35rem 0">Why this professor</p>'
                card += f'<p style="font-size:0.88rem;color:#aaa;line-height:1.65;margin:0 0 0.75rem 0">{reason}</p>'

                # New directions
                if gap_tags:
                    card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#444;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.35rem 0">New research directions</p>'
                    card += gap_tags

                # Open problems
                if fw_text:
                    card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#444;text-transform:uppercase;letter-spacing:0.12em;margin:0.75rem 0 0.35rem 0">Open problems</p>'
                    card += f'<p style="font-size:0.82rem;color:#666;line-height:1.6;font-style:italic;margin:0 0 0.75rem 0">{fw_text}</p>'

                # Email
                if i <= 2:
                    email_content = email if email else hook
                    if email_content:
                        label = "Drafted email" if email else "Email opening"
                        card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#444;text-transform:uppercase;letter-spacing:0.12em;margin:0.75rem 0 0.35rem 0">{label}</p>'
                        card += f'<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:2px;padding:1.1rem 1.3rem;font-size:0.86rem;color:#aaa;line-height:1.7;white-space:pre-wrap;font-family:IBM Plex Sans,sans-serif">{email_content}</div>'

                # Selection note
                card += f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#2e2e2e;border-left:2px solid #1e1e1e;padding-left:0.75rem;margin:1.2rem 0 0 0;line-height:1.6">{note}</p>'

                card += '</div>'

                st.markdown(card, unsafe_allow_html=True)