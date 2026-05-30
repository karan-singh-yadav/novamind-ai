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
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* GOOGLE FONT */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* GLOBAL */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* APP BACKGROUND */

.stApp {
    background:
    radial-gradient(circle at top left, rgba(59,130,246,0.15), transparent 30%),
    radial-gradient(circle at bottom right, rgba(168,85,247,0.15), transparent 30%),
    #050816;

    color: white;
}

/* REMOVE STREAMLIT DEFAULT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* MAIN CONTAINER */

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HERO SECTION */

.hero {
    background:
    linear-gradient(
        135deg,
        rgba(59,130,246,0.15),
        rgba(168,85,247,0.15)
    );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 30px;

    padding: 60px;

    position: relative;

    overflow: hidden;

    backdrop-filter: blur(20px);

    margin-bottom: 35px;
}

/* GLOW */

.hero::before {
    content: "";

    position: absolute;

    width: 400px;
    height: 400px;

    background: rgba(59,130,246,0.2);

    filter: blur(120px);

    top: -150px;
    right: -150px;
}

/* TITLE */

.main-title {

    font-size: 4.5rem;

    font-weight: 800;

    line-height: 1.1;

    margin-bottom: 15px;

    background: linear-gradient(
        90deg,
        #60A5FA,
        #A78BFA,
        #F472B6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */

.subtitle {

    color: #CBD5E1;

    font-size: 1.15rem;

    max-width: 700px;
}

/* FEATURE CARDS */

.feature-card {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    transition: 0.3s ease;

    backdrop-filter: blur(12px);

    height: 100%;
}

.feature-card:hover {

    transform: translateY(-6px);

    border: 1px solid rgba(96,165,250,0.4);

    box-shadow: 0 20px 40px rgba(59,130,246,0.12);
}

/* UPLOAD CONTAINER */

.upload-box {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 28px;

    padding: 28px;

    margin-top: 30px;

    margin-bottom: 30px;

    backdrop-filter: blur(15px);
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.03);

    border: 2px dashed rgba(96,165,250,0.5);

    border-radius: 22px;

    padding: 20px;
}

/* FILE TEXT */

[data-testid="stFileUploader"] * {

    color: white !important;

    font-size: 16px !important;
}

/* DROPZONE TEXT */

[data-testid="stFileUploaderDropzoneInstructions"] span {

    color: white !important;

    font-size: 18px !important;

    font-weight: 600 !important;
}

/* SMALL TEXT */

[data-testid="stFileUploaderDropzone"] small {

    color: #CBD5E1 !important;
}

/* CHAT MESSAGE */

.stChatMessage {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 22px;

    padding: 18px;

    backdrop-filter: blur(10px);

    margin-bottom: 14px;
}

/* CHAT INPUT */

.stChatInputContainer {

    background: rgba(255,255,255,0.06) !important;

    border-radius: 18px !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    padding: 8px;
}

/* INPUT TEXT */

textarea {

    color: white !important;

    font-size: 16px !important;
}

/* PLACEHOLDER */

textarea::placeholder {

    color: #CBD5E1 !important;
}

/* SUCCESS */

.stSuccess {

    background: rgba(16,185,129,0.15) !important;

    color: #D1FAE5 !important;

    border-radius: 14px;
}

/* TEXT */

p, span, label, div {

    color: white;
}

/* CHAT ICONS */

[data-testid="stChatMessageAvatarUser"] {
    background: #2563EB;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #7C3AED;
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #0F172A;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">

<div class="main-title">
NovaMind AI
</div>

<div class="subtitle">
A modern AI-powered PDF assistant using Retrieval-Augmented Generation architecture with semantic search and ultra-fast inference.
</div>

</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE SECTION ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h2>⚡ Lightning Fast</h2>
    <p>Powered by Groq for ultra-fast AI responses and smooth interactions.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h2>🧠 Semantic Search</h2>
    <p>Find relevant answers instantly using vector embeddings and AI retrieval.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
    <h2>📚 Smart PDF Chat</h2>
    <p>Upload documents and interact with them naturally using conversational AI.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------

st.markdown("""
<div class="upload-box">

<h2 style="
font-size: 1.8rem;
font-weight: 700;
margin-bottom: 10px;
">
📄 Upload Your PDF
</h2>

<p style="
color: #CBD5E1;
font-size: 1rem;
margin-bottom: 18px;
">
Upload a PDF document and start chatting with AI instantly.
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type="pdf",
    label_visibility="collapsed"
)

# ---------------- PDF PROCESSING ----------------

if uploaded_file:

    with st.spinner("⚡ Processing your document..."):

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = load_pdf(uploaded_file.name)

        chunks = split_text(text)

        store_embeddings(chunks)

    st.success("✅ PDF processed successfully!")

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------

query = st.chat_input(
    "💬 Ask something from your document..."
)

# ---------------- AI RESPONSE ----------------

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("🧠 AI is thinking..."):

            docs = retrieve_docs(query)

            context = "\\n".join(docs)

            response = get_llm_response(
                query,
                context
            )

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ---------------- FOOTER ----------------

st.write("")

st.markdown("""

<center>

<p style='color:#94A3B8;'>

Built with Streamlit • Groq • ChromaDB • LangChain

</p>

</center>

""", unsafe_allow_html=True)

