import os
import tempfile
import streamlit as st
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_splitter import split_text
from src.retrieval.vector_store import store_embeddings
from src.retrieval.retriever import retrieve_docs
from src.llm.grok_model import get_llm_response

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NovaMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ---- APP BG ---- */
.stApp {
    background: #09090F;
    color: #E2E8F0;
    min-height: 100vh;
}

/* Noise grain overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.35;
}

/* Remove streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1280px;
    padding: 2.5rem 2rem 4rem;
    position: relative;
    z-index: 1;
}

/* ---- AMBIENT GLOW ---- */
.ambient-left {
    position: fixed;
    top: -200px;
    left: -200px;
    width: 700px;
    height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.ambient-right {
    position: fixed;
    bottom: -200px;
    right: -200px;
    width: 700px;
    height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ---- TOP NAV BAR ---- */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 2.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.navbar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 24px rgba(59,130,246,0.35);
}
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.nav-badge {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60A5FA;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---- HERO ---- */
.hero-section {
    margin-bottom: 3.5rem;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.25);
    color: #A78BFA;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hero-eyebrow-dot {
    width: 6px;
    height: 6px;
    background: #A78BFA;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
.hero-heading {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 5vw, 4.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin-bottom: 1.2rem;
    color: #F1F5F9;
}
.hero-heading span {
    background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #64748B;
    font-size: 1.1rem;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.7;
}

/* ---- STATS ROW ---- */
.stats-row {
    display: flex;
    gap: 32px;
    margin-top: 2.2rem;
    flex-wrap: wrap;
}
.stat-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.02em;
}
.stat-label {
    font-size: 0.75rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}
.stat-divider {
    width: 1px;
    background: rgba(255,255,255,0.07);
    align-self: stretch;
}

/* ---- CAPABILITY CARDS ---- */
.cap-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 3rem;
}
.cap-card {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: default;
}
.cap-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.04), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.cap-card:hover {
    border-color: rgba(96,165,250,0.3);
    transform: translateY(-4px);
    box-shadow: 0 20px 48px rgba(0,0,0,0.4), 0 0 0 1px rgba(96,165,250,0.1);
}
.cap-card:hover::after { opacity: 1; }

.cap-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 14px;
}
.cap-icon-blue { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.2); }
.cap-icon-purple { background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.2); }
.cap-icon-pink { background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.2); }

.cap-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 6px;
}
.cap-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.6;
    font-weight: 300;
}
.cap-tag {
    display: inline-block;
    margin-top: 14px;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 100px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.tag-blue { background: rgba(59,130,246,0.1); color: #60A5FA; border: 1px solid rgba(59,130,246,0.2); }
.tag-purple { background: rgba(139,92,246,0.1); color: #A78BFA; border: 1px solid rgba(139,92,246,0.2); }
.tag-pink { background: rgba(236,72,153,0.1); color: #F472B6; border: 1px solid rgba(236,72,153,0.2); }

/* ---- DIVIDER ---- */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 2.5rem 0;
}

/* ---- UPLOAD PANEL ---- */
.upload-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 32px 36px;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.upload-panel::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(96,165,250,0.4), transparent);
}
.upload-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.upload-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #E2E8F0;
}
.upload-subtitle {
    color: #475569;
    font-size: 0.9rem;
    font-weight: 300;
    margin-bottom: 22px;
    line-height: 1.6;
}
.step-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(59,130,246,0.3);
}

/* ---- FILE UPLOADER STYLE ---- */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(96,165,250,0.3) !important;
    border-radius: 18px !important;
    padding: 12px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.6) !important;
}
[data-testid="stFileUploader"] * { color: #CBD5E1 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
}
[data-testid="stFileUploaderDropzone"] small { color: #475569 !important; }

/* ---- CHAT SECTION HEADER ---- */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.chat-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.chat-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #10B981;
    font-weight: 500;
}
.chat-status-dot {
    width: 7px;
    height: 7px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse-green 2s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(16,185,129,0.5);
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- CHAT MESSAGES ---- */
.stChatMessage {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
}
.stChatMessage p { color: #CBD5E1 !important; line-height: 1.7; }

[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    border-radius: 10px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #6D28D9, #8B5CF6) !important;
    border-radius: 10px !important;
}

/* ---- CHAT INPUT ---- */
.stChatInputContainer {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: border-color 0.3s !important;
}
.stChatInputContainer:focus-within {
    border-color: rgba(96,165,250,0.35) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
}
textarea {
    color: #E2E8F0 !important;
    font-size: 0.95rem !important;
    font-family: 'DM Sans', sans-serif !important;
    background: transparent !important;
}
textarea::placeholder { color: #475569 !important; }

/* ---- SUCCESS / INFO ---- */
.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    color: #6EE7B7 !important;
    border-radius: 14px !important;
}
.stSuccess * { color: #6EE7B7 !important; }

.stSpinner * { color: #60A5FA !important; }

/* ---- EMPTY STATE ---- */
.empty-chat {
    text-align: center;
    padding: 3.5rem 1rem;
    color: #334155;
}
.empty-chat-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-chat-text {
    font-size: 0.9rem;
    color: #334155;
    font-weight: 300;
    line-height: 1.7;
}

/* ---- FOOTER ---- */
.footer {
    text-align: center;
    padding-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 2rem;
}
.footer-tech {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
}
.footer-chip {
    font-size: 0.72rem;
    font-weight: 500;
    color: #334155;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.footer-dot { color: #1E293B; }

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---- SPINNER ---- */
[data-testid="stSpinner"] { color: #60A5FA !important; }

p, span, label, div { color: inherit; }

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- AMBIENT GLOWS ----------------

st.markdown(
    """
<div class="ambient-left"></div>
<div class="ambient-right"></div>
""",
    unsafe_allow_html=True,
)

# ---------------- NAVBAR ----------------

st.markdown(
    """
<div class="navbar">
    <div class="navbar-logo">
        <div class="logo-icon">🧠</div>
        <div class="logo-text">NovaMind AI</div>
    </div>
    <div class="nav-badge">✦ RAG Powered</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- HERO ----------------

st.markdown(
    """
<div class="hero-section">
    <div class="hero-eyebrow">
        <div class="hero-eyebrow-dot"></div>
        AI-Powered Document Intelligence
    </div>
    <div class="hero-heading">
        Chat with your PDFs<br>
        <span>the intelligent way.</span>
    </div>
    <div class="hero-sub">
        Upload any document and instantly query it using semantic search, vector embeddings, and ultra-fast LLM inference.
    </div>
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-val">&lt; 1s</div>
            <div class="stat-label">Response time</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-val">RAG</div>
            <div class="stat-label">Architecture</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-val">∞</div>
            <div class="stat-label">Context aware</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- CAPABILITY CARDS ----------------

st.markdown(
    """
<div class="cap-grid">
    <div class="cap-card">
        <div class="cap-icon cap-icon-blue">⚡</div>
        <div class="cap-title">Lightning Fast</div>
        <div class="cap-desc">Groq-powered inference delivers sub-second responses with no lag or wait time.</div>
        <span class="cap-tag tag-blue">Groq LPU</span>
    </div>
    <div class="cap-card">
        <div class="cap-icon cap-icon-purple">🔍</div>
        <div class="cap-title">Semantic Search</div>
        <div class="cap-desc">ChromaDB vector store finds the most relevant document chunks with precision.</div>
        <span class="cap-tag tag-purple">ChromaDB</span>
    </div>
    <div class="cap-card">
        <div class="cap-icon cap-icon-pink">📄</div>
        <div class="cap-title">Smart PDF Parsing</div>
        <div class="cap-desc">LangChain-powered text splitting preserves semantic meaning across all chunks.</div>
        <span class="cap-tag tag-pink">LangChain</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- DIVIDER ----------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------- UPLOAD PANEL ----------------

st.markdown(
    """
<div class="upload-panel">
    <div class="upload-header">
        <div class="step-number">01</div>
        <div class="upload-title">Upload Your Document</div>
    </div>
    <div class="upload-subtitle">
        Drop a PDF below — it will be parsed, chunked, and embedded into the vector store automatically.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

# ---------------- PDF PROCESSING ----------------

if uploaded_file:
    temp_path = None
    with st.spinner("⚡ Parsing and embedding your document..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getbuffer())
                temp_path = temp_pdf.name

            text = load_pdf(temp_path)
            chunks = split_text(text)

            if not chunks:
                st.error("The uploaded PDF did not contain readable text.")
            else:
                store_embeddings(chunks)
                st.session_state.pdf_ready = True
                st.success(f"✅ **{uploaded_file.name}** — Ready to chat")
        except Exception as err:
            st.error("Unable to process the uploaded PDF. Please try a different file.")
            st.exception(err)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

# ---------------- DIVIDER ----------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------- CHAT SECTION ----------------

status_html = ""
if st.session_state.pdf_ready:
    status_html = '<div class="chat-status"><div class="chat-status-dot"></div> Document loaded</div>'
else:
    status_html = '<div class="chat-status" style="color:#475569;"><span style="color:#334155;">○</span>&nbsp; Awaiting document</div>'

st.markdown(
    f"""
<div class="chat-header">
    <div class="chat-title">
        <span>💬</span> Conversation
    </div>
    {status_html}
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- EMPTY STATE ----------------

if not st.session_state.messages:
    st.markdown(
        """
    <div class="empty-chat">
        <div class="empty-chat-icon">🧠</div>
        <div class="empty-chat-text">
            No messages yet.<br>
            Upload a PDF above, then ask your first question below.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------

query = st.chat_input("Ask something from your document…")

# ---------------- AI RESPONSE ----------------

if query:
    if not st.session_state.pdf_ready:
        st.warning("Upload a PDF first before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                docs = retrieve_docs(query)
                context = "\n".join(docs)
                response = get_llm_response(query, context)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------- FOOTER ----------------

st.markdown(
    """
<div class="footer">
    <div class="footer-tech">
        <span class="footer-chip">Streamlit</span>
        <span class="footer-chip">Groq</span>
        <span class="footer-chip">ChromaDB</span>
        <span class="footer-chip">LangChain</span>
        <span class="footer-chip">RAG</span>
    </div>
    <p style="margin-top: 16px; font-size: 0.72rem; color: #1E293B; font-weight: 300;">
        NovaMind AI — Document Intelligence Platform
    </p>
</div>
""",
    unsafe_allow_html=True,
)
