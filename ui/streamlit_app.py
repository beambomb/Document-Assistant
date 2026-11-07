"""Streamlit Interactive Web UI for PT Majelis Debu Bintang.

Minimalist, clean, anti-slop interface inspired by Claude:
- Chat input firmly at the bottom of the conversational thread
- User chat bubbles aligned to the RIGHT with avatar 👤
- AI assistant chat bubbles aligned to the LEFT with avatar 🤖
- Persistent chat history in session state
- Emojis restricted strictly to user (👤) and assistant (🤖) avatars
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import html
import time
from datetime import datetime
import streamlit as st
import pandas as pd

from src.core.models import UserContext
from src.ingestion.loader import DocumentLoader
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import DenseVectorIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.confidence_gate import ConfidenceGate
from src.generation.generator import get_generator
from src.generation.citation_verifier import CitationVerifier
from src.evaluation.harness import EvaluationHarness
from src.evaluation.reporter import EvaluationReporter
from config.settings import settings

# Page setup
st.set_page_config(
    page_title="PT Majelis Debu Bintang",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Minimalist CSS - Claude-Style Conversational Canvas
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #1A1917;
        --sidebar: #151413;
        --surface: #242321;
        --surface-user: #2C2A28;
        --border: #33312E;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --text: #FAF7F2;
        --text-muted: #8E8980;
        --terracotta: #D97757;
        --terracotta-hover: #E28668;
    }

    /* Base Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
        background-color: var(--bg);
        font-size: 15px;
    }

    h1, h2, h3, .serif-brand {
        font-family: 'Newsreader', Georgia, serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
    }

    /* Streamlit Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: var(--bg) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        height: 3rem !important;
        z-index: 99 !important;
    }

    /* Container width & vertical breathing room with EQUAL EDGES & NO TOP CLIPPING */
    .main .block-container {
        padding-top: 4.75rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    /* Segmented Control Pill Navigation */
    div[data-testid="stSegmentedControl"] {
        margin-bottom: 1.75rem !important;
    }
    div[data-testid="stSegmentedControl"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background: transparent !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 6px 18px !important;
        color: var(--text-muted) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color: var(--text) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: #363431 !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
    }

    /* Clean Pill Tabs (Zero clipping, zero underlines) */
    div[data-testid="stTabs"] div[role="tablist"],
    div[data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 1.75rem !important;
        width: fit-content !important;
        border-bottom: none !important;
    }
    div[data-testid="stTabs"] div[role="tablist"]::after,
    div[data-testid="stTabs"] hr,
    div[data-baseweb="tab-highlight"], 
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
    button[data-testid="stTab"],
    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 6px 18px !important;
        color: var(--text-muted) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        height: auto !important;
        transition: all 0.15s ease !important;
    }
    button[data-testid="stTab"]:hover,
    button[data-baseweb="tab"]:hover {
        color: var(--text) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    button[data-testid="stTab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #363431 !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
        border-bottom: none !important;
    }

    /* Minimal Prompt Suggestion Pills (Scoped strictly to prompt-pills) */
    div.prompt-pills div.stHorizontalBlock div[data-testid="stButton"] button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 5px 14px !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
        box-shadow: none !important;
    }
    div.prompt-pills div.stHorizontalBlock div[data-testid="stButton"] button:hover {
        border-color: var(--terracotta) !important;
        color: var(--text) !important;
        transform: translateY(-1px) !important;
        background: var(--surface) !important;
    }

    /* Native Bilateral Chat Layout with Equal Edges */
    div[data-testid="stChatMessage"] {
        padding: 0.85rem 0 !important;
        gap: 0.85rem !important;
        border: none !important;
        background: transparent !important;
        width: 100% !important;
    }

    /* User Message: Aligned to the RIGHT with Terracotta Tint */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"][aria-label*="user"] {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 80% !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"][aria-label*="user"] div[data-testid="stChatMessageContent"] {
        background: var(--surface-user) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px 4px 16px 16px !important;
        padding: 10px 16px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2) !important;
        text-align: left !important;
        color: var(--text) !important;
    }

    /* Assistant Message: Aligned to the LEFT with Clean Typography */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"][aria-label*="assistant"] {
        flex-direction: row !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 100% !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"][aria-label*="assistant"] div[data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        padding: 2px 0 !important;
        color: var(--text) !important;
        line-height: 1.65 !important;
    }

    /* Avatar styling */
    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: transparent !important;
        border: none !important;
        font-size: 1.35rem !important;
        margin-top: 2px !important;
    }

    /* Citations */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(217, 119, 87, 0.12);
        border: 1px solid rgba(217, 119, 87, 0.28);
        color: #E28668;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Refusal Callout Box */
    .abstain-box {
        background: var(--surface);
        border-left: 3px solid var(--terracotta);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
        color: var(--text);
        margin: 4px 0;
    }

    /* Bottom Docked Chat Input (Streamlit native stBottom container) */
    div[data-testid="stBottom"] {
        background-color: var(--bg) !important;
        border-top: 1px solid var(--border-subtle) !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }
    div[data-testid="stBottom"] > div {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 0 !important;
        padding-bottom: 2px !important;
    }
    div[data-testid="stChatInput"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stBottom"] footer,
    footer[data-testid="stFooter"],
    footer {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stChatInput"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 2px rgba(217, 119, 87, 0.2) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.94rem !important;
        color: var(--text) !important;
    }
    div[data-testid="stChatInput"] button {
        background: var(--terracotta) !important;
        color: white !important;
        border-radius: 50% !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background: var(--terracotta-hover) !important;
    }

    /* Quietly hide bottom chat bar when viewing Benchmark or Dokumen tabs */
    body:has(button[data-testid="stTab"]:nth-child(2)[aria-selected="true"]) div[data-testid="stBottom"],
    body:has(button[data-testid="stTab"]:nth-child(3)[aria-selected="true"]) div[data-testid="stBottom"],
    body:has(button[role="tab"]:nth-child(2)[aria-selected="true"]) div[data-testid="stBottom"],
    body:has(button[role="tab"]:nth-child(3)[aria-selected="true"]) div[data-testid="stBottom"] {
        display: none !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem !important;
    }

    /* Clean Radio Options */
    div[data-testid="stRadio"] > div {
        gap: 4px !important;
    }
    div[data-testid="stRadio"] label {
        font-size: 0.88rem !important;
        color: var(--text-muted) !important;
        padding: 3px 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: var(--text) !important;
    }

    /* Clean Selectboxes, Text Inputs, and Sliders in Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] label,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] label {
        font-size: 0.78rem !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.84rem !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.84rem !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        margin: 6px 0 !important;
    }

    /* Action & Primary Buttons (Terracotta Orange) */
    div[data-testid="stButton"] > button,
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    div.stHorizontalBlock div[data-testid="stButton"] button[kind="primary"],
    div.stHorizontalBlock div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
        background: var(--terracotta) !important;
        background-color: var(--terracotta) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        padding: 7px 18px !important;
        box-shadow: 0 1px 4px rgba(217, 119, 87, 0.28) !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stButton"] > button:hover,
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    div.stHorizontalBlock div[data-testid="stButton"] button[kind="primary"]:hover,
    div.stHorizontalBlock div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
        background: var(--terracotta-hover) !important;
        background-color: var(--terracotta-hover) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(217, 119, 87, 0.45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_engine():
    """Initializes corpus and indexing engines."""
    loader = DocumentLoader()
    corpus_dir = settings.raw_data_dir / "id_enterprise"
    docs, chunks = loader.load_directory_markdown(corpus_dir)

    bm25 = BM25Index()
    bm25.build(chunks)

    vector_idx = DenseVectorIndex(use_mock=True)
    vector_idx.build(chunks)

    retriever = HybridRetriever(bm25_index=bm25, vector_index=vector_idx)
    reranker = CrossEncoderReranker(use_mock=True)
    gate = ConfidenceGate(threshold=settings.confidence_threshold)
    generator = get_generator()
    verifier = CitationVerifier()

    return docs, chunks, bm25, vector_idx, retriever, reranker, gate, generator, verifier


docs, chunks, bm25, vector_idx, retriever, reranker, gate, generator, verifier = load_rag_engine()

# Initialize session state for persistent chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: Access Control (Clean & Minimalist) ---
with st.sidebar:
    st.markdown('<div class="serif-brand" style="font-size: 1.35rem; margin-bottom: 2px;">PT Majelis Debu Bintang</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.8rem; color: #8E8980; margin-bottom: 1.5rem;">Asisten Dokumen & Pengetahuan</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.78rem; font-weight: 600; color: #FAF7F2; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Peran Aktif</div>', unsafe_allow_html=True)

    role_options = {
        "guest": "Tamu",
        "employee": "Karyawan",
        "finance": "Keuangan",
        "hr": "HR & Personalia",
        "management": "Direksi",
        "admin": "Admin",
    }

    selected_role = st.radio(
        "Peran Pengguna",
        options=list(role_options.keys()),
        format_func=lambda r: role_options[r],
        index=1,
        label_visibility="collapsed",
    )

    if st.button("Bersihkan Obrolan", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown('<div style="font-size: 0.78rem; font-weight: 600; color: #FAF7F2; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Pilihan Model AI</div>', unsafe_allow_html=True)

    provider_options = {
        "openrouter": "Cloud AI (OpenRouter)",
        "gguf": "Lokal CPU (Offline GGUF)",
    }
    selected_provider = st.radio(
        "Penyedia AI",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=0 if settings.llm_provider.lower() == "openrouter" else 1,
        label_visibility="collapsed",
        key="selected_provider",
    )

    if selected_provider == "openrouter":
        cloud_models = {
            "meta-llama/llama-3.3-70b-instruct": "Meta Llama 3.3 70B (Rekomendasi)",
            "deepseek/deepseek-chat": "DeepSeek V3 (Sangat Cerdas)",
            "google/gemini-2.0-flash-exp:free": "Google Gemini 2.0 Flash (Gratis)",
            "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B (Multibahasa)",
            "openai/gpt-4o-mini": "OpenAI GPT-4o Mini (Cepat)",
            "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet (Penalaran)",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "Nemotron 30B (Penalaran)",
            "custom": "Model Kustom (Input Slug...)",
        }
        selected_cloud_key = st.selectbox(
            "Model Cloud",
            options=list(cloud_models.keys()),
            format_func=lambda k: cloud_models[k],
            index=0,
            key="selected_cloud_key",
        )
        if selected_cloud_key == "custom":
            active_cloud_model = st.text_input(
                "Slug Model OpenRouter",
                value="mistralai/mistral-small-3",
                placeholder="contoh: meta-llama/llama-3.1-8b-instruct",
                key="custom_cloud_slug",
            )
        else:
            active_cloud_model = selected_cloud_key

        with st.expander("API Key Kustom (BYOK)", expanded=False):
            user_api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                placeholder="sk-or-v1-...",
                help="Biarkan kosong untuk menggunakan kunci default sistem",
                key="user_api_key",
            )
        active_api_key = user_api_key.strip() if user_api_key.strip() else settings.openrouter_api_key
        active_model_name = active_cloud_model
        model_display = cloud_models.get(active_cloud_model, active_cloud_model.split("/")[-1])
        engine_display = "OpenRouter (Cloud)"

    else:  # gguf local
        local_model_files = sorted(list(Path("models").glob("*.gguf")))
        if local_model_files:
            def format_gguf_label(p: Path) -> str:
                size_mb = p.stat().st_size / (1024 * 1024)
                name = p.name
                if "qwen2.5-0.5b" in name.lower():
                    return f"Qwen 2.5 0.5B ({size_mb:.0f} MB)"
                return f"{name} ({size_mb:.0f} MB)"

            selected_local_file = st.selectbox(
                "File Model GGUF",
                options=local_model_files,
                format_func=format_gguf_label,
                index=0,
                key="selected_local_file",
            )
            active_local_model_path = str(selected_local_file)
            size_mb = selected_local_file.stat().st_size / (1024 * 1024)
            st.caption(f"Status: Siap di CPU ({size_mb:.0f} MB)")
        else:
            st.warning("Tidak ada file .gguf di folder models/.")
            active_local_model_path = settings.local_model_path

        active_model_name = active_local_model_path
        model_display = Path(active_local_model_path).stem
        engine_display = "GGUF (Lokal CPU)"

    # Advanced Generation Settings Expander
    with st.expander("Pengaturan Inferensi", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2 if selected_provider == "openrouter" else 0.3,
            step=0.05,
            help="0.0 = deterministik dan patuh konteks; 1.0 = lebih variatif",
            key="slider_temperature",
        )
        max_tokens = st.slider(
            "Maksimal Token Jawaban",
            min_value=100,
            max_value=800,
            value=350,
            step=50,
            help="Batas panjang jawaban AI",
            key="slider_max_tokens",
        )
        if selected_provider == "gguf":
            repeat_penalty = st.slider(
                "Anti Pengulangan (Penalty)",
                min_value=1.0,
                max_value=1.5,
                value=1.22,
                step=0.02,
                help="Mencegah model kecil mengulang-ulang kalimat",
                key="slider_repeat_penalty",
            )
            n_threads = st.selectbox(
                "CPU Threads",
                options=[2, 4, 6, 8],
                index=1,
                help="Jumlah core CPU yang digunakan",
                key="select_cpu_threads",
            )
        else:
            repeat_penalty = 1.0
            n_threads = 4

    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: #8E8980; line-height: 1.7; border-top: 1px solid var(--border-subtle); padding-top: 10px; margin-top: 10px;">
            <div><strong>Engine:</strong> {engine_display}</div>
            <div><strong>Model:</strong> {model_display}</div>
            <div><strong>Indeks:</strong> {len(docs)} Dokumen &middot; {len(chunks)} Chunks</div>
            <div><strong>Gate:</strong> {settings.confidence_threshold} &middot; <strong>Temp:</strong> {temperature}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- MODAL: TAMBAH DOKUMEN RESMI ---
@st.dialog("Tambah Dokumen Resmi")
def modal_tambah_dokumen():
    st.caption("Dokumen baru akan otomatis dipecah menjadi chunks dan diindeks secara live ke dalam BM25 & Dense Vector Index.")

    up_col1, up_col2 = st.columns(2)
    new_doc_id = up_col1.text_input("Nomor Dokumen", placeholder="misal: SOP-OPS-006", key="modal_doc_id")
    new_title = up_col2.text_input("Judul Dokumen", placeholder="misal: SOP Pengelolaan Fasilitas Kantor", key="modal_doc_title")

    up_col3, up_col4 = st.columns(2)
    dept_options = ["Umum", "Operasional", "IT", "HR", "Keuangan", "Legal"]
    new_department = up_col3.selectbox("Departemen / Divisi", options=dept_options, key="modal_dept")
    new_roles = up_col4.multiselect(
        "Klasifikasi Hak Akses",
        options=["guest", "employee", "finance", "hr", "management", "admin"],
        default=["employee"],
        key="modal_roles",
        help="Peran pengguna mana saja yang diizinkan mengakses dokumen ini.",
    )

    uploaded_doc = st.file_uploader("Upload Dokumen (.md / .txt)", type=["md", "txt"], key="modal_upload_file")
    default_body = ""
    if uploaded_doc is not None:
        try:
            default_body = uploaded_doc.read().decode("utf-8")
        except Exception:
            default_body = ""

    new_content = st.text_area(
        "Isi / Konten Dokumen (Markdown)",
        value=default_body,
        height=180,
        placeholder="Tuliskan isi SOP, kebijakan, atau aturan operasional resmi di sini...",
        key="modal_content",
    )

    if st.button("Simpan & Indeks Dokumen", type="primary", use_container_width=True, key="modal_btn_save"):
        if not new_doc_id or not new_title or not new_content.strip():
            st.error("Mohon lengkapi Nomor Dokumen, Judul Dokumen, dan Isi Konten.")
        else:
            dept_folder = settings.raw_data_dir / "id_enterprise" / new_department.lower().replace(" ", "_")
            dept_folder.mkdir(parents=True, exist_ok=True)
            clean_filename = f"{new_doc_id.strip().upper()}.md"
            target_filepath = dept_folder / clean_filename

            role_str = ", ".join(new_roles) if new_roles else "employee"
            today_str = datetime.now().strftime("%Y-%m-%d")

            md_header = f"""# {new_title.strip()}

**Nomor Dokumen**: {new_doc_id.strip().upper()}  
**Versi**: 1.0  
**Tanggal Berlaku**: {today_str}  
**Klasifikasi Akses**: {role_str}  
**Divisi**: {new_department}  

---

"""
            final_text = new_content.strip()
            if not final_text.startswith("#"):
                final_text = md_header + final_text

            with open(target_filepath, "w", encoding="utf-8") as f:
                f.write(final_text)

            load_rag_engine.clear()
            st.success(f"Dokumen {new_doc_id.strip().upper()} berhasil disimpan dan diindeks ulang!")
            time.sleep(0.6)
            st.rerun()


# --- MAIN WORKSPACE PILL NAVIGATION ---
quick_query = None
active_tab = st.segmented_control(
    "Navigasi",
    options=["Percakapan", "Benchmark", "Dokumen"],
    default="Percakapan",
    label_visibility="collapsed",
)

if active_tab == "Percakapan":
    # Clean, Minimalist Welcome Header
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; margin: 0 0 6px 0;">PT Majelis Debu Bintang</h1>
            <div style="color: #8E8980; font-size: 0.95rem;">
                Asisten dokumen resmi berbasis peran (RBAC) dengan sitasi terverifikasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sleek Horizontal Prompt Pills
    st.markdown('<div class="prompt-pills">', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)

    if p1.button("Sandi Laptop IT"):
        quick_query = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"
    if p2.button("Hak Cuti Tahunan"):
        quick_query = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"
    if p3.button("Uji ACL (Gaji)"):
        quick_query = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"
    if p4.button("Uji Abstention"):
        quick_query = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RENDER PERSISTENT CHAT HISTORY (User on RIGHT, Assistant on LEFT) ---
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    role_tag = msg.get("role_label", "Pengguna")
                    st.markdown(
                        f"<div style='font-size: 0.72rem; color: #D97757; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;'>{role_tag}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    msg_model = msg.get("model", "")
                    model_str = f" &middot; {msg_model}" if msg_model else ""
                    if msg.get("abstained"):
                        st.markdown(
                            f"<div class='abstain-box'>{html.escape(msg.get('content', ''))}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Latensi: {msg.get('latency_ms', 0):.1f} ms &middot; Keyakinan: {msg.get('score', 0):.3f} &middot; Status: Ditolak{model_str}"
                        )
                    else:
                        st.markdown(msg.get("content", ""))
                        citations = msg.get("citations", [])
                        if citations:
                            pills = "".join([
                                f"<span class='citation-tag'>{c['doc_id']} &middot; {c['section']} (Hal. {c['page']})</span>"
                                for c in citations
                            ])
                            st.markdown(f"<div style='margin-top: 6px;'>{pills}</div>", unsafe_allow_html=True)
                        st.caption(
                            f"Latensi: {msg.get('latency_ms', 0):.1f} ms &middot; Keyakinan: {msg.get('score', 0):.3f}{model_str}"
                        )
    else:
        st.markdown(
            """
            <div style="padding: 3rem 0; text-align: center; color: #8E8980; font-size: 0.9rem;">
                Belum ada percakapan. Ketik pertanyaan di bawah atau pilih pertanyaan uji cepat di atas.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- VIEWPORT-LOCKED BOTTOM CHAT INPUT (ONLY ON PERCAKAPAN TAB) ---
    query_input = st.chat_input("Tanyakan sesuatu tentang dokumen organisasi...")
    active_query = quick_query or query_input

    if active_query:
        # Append User Message immediately to session state
        st.session_state.chat_history.append({
            "role": "user",
            "content": active_query,
            "role_label": role_options[selected_role],
        })

        user = UserContext(user_id="user_active", roles=[selected_role])
        t0 = time.perf_counter()

        # Step 1: Hybrid Retrieval with ACL pre-filtering
        fused = retriever.search(active_query, user, top_k=settings.top_k_fused)

        # Step 2: Cross-Encoder Reranking
        reranked = reranker.rerank(active_query, fused, top_k=settings.top_k_final)

        # Step 3: Confidence Gating
        should_abstain, refusal_reason, conf_score = gate.evaluate(reranked)

        latency_ms = (time.perf_counter() - t0) * 1000

        if should_abstain:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": refusal_reason,
                "abstained": True,
                "score": conf_score,
                "latency_ms": latency_ms,
                "model": "Abstain (Gate)",
            })
        else:
            if selected_provider == "gguf":
                active_gen = get_generator(
                    provider="gguf",
                    model=active_local_model_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_threads=n_threads,
                    repeat_penalty=repeat_penalty,
                )
                chosen_tag = "Qwen 2.5 0.5B (Lokal)" if "qwen" in active_local_model_path.lower() else f"{Path(active_local_model_path).stem} (Lokal)"
            else:
                active_gen = get_generator(
                    provider="openrouter",
                    model=active_cloud_model,
                    api_key=active_api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                chosen_tag = cloud_models.get(active_cloud_model, active_cloud_model.split("/")[-1])

            answer = active_gen.generate(active_query, reranked, [selected_role])
            citations, precision, _ = verifier.extract_and_verify(answer, reranked)

            cit_dicts = [
                {"doc_id": c.doc_id, "section": c.section, "page": c.page}
                for c in citations
            ]

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "abstained": False,
                "citations": cit_dicts,
                "score": conf_score,
                "latency_ms": latency_ms,
                "model": chosen_tag,
            })

        st.rerun()

elif active_tab == "Benchmark":
    st.markdown('<h2 style="font-size: 1.6rem; margin-bottom: 2px;">Evaluasi & Benchmark</h2>', unsafe_allow_html=True)
    st.caption("Uji kuantitatif Information Retrieval, Keamanan Hak Akses (ACL), dan Abstention.")

    if st.button("Jalankan Evaluasi Ulang", type="primary"):
        with st.spinner("Menjalankan evaluasi kuantitatif..."):
            harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
            results = harness.run_ablation()
            EvaluationReporter.save_report(
                rows=results,
                output_md_path=Path("reports/ablation_results.md"),
                output_csv_path=Path("reports/ablation_results.csv"),
            )
            st.success("Evaluasi selesai.")

    # Render benchmark table directly and prominently on screen
    csv_file = Path("reports/ablation_results.csv")
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            display_df = pd.DataFrame({
                "Konfigurasi": df["mode"],
                "Recall@1": df["recall_at_1"].map(lambda x: f"{x*100:.1f}%"),
                "Recall@3": df["recall_at_3"].map(lambda x: f"{x*100:.1f}%"),
                "Recall@5": df["recall_at_5"].map(lambda x: f"{x*100:.1f}%"),
                "MRR": df["mrr"].map(lambda x: f"{x:.3f}"),
                "nDCG@5": df["ndcg_at_5"].map(lambda x: f"{x:.3f}"),
                "Kebocoran ACL": df["leakage_rate_pct"].map(lambda x: f"{x:.2f}%"),
                "Penolakan Jebakan": df["trap_rejection_pct"].map(lambda x: f"{x:.1f}%"),
                "Latensi": df["latency_ms"].map(lambda x: f"{x:.1f} ms"),
            })
            st.write("")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        except Exception:
            pass

    # Narrative analysis in clean quiet expander
    report_file = Path("reports/ablation_results.md")
    if report_file.exists():
        with st.expander("Ringkasan Metrik & Analisis Temuan"):
            with open(report_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())

elif active_tab == "Dokumen":
    head_col, action_col = st.columns([3, 1], vertical_alignment="center")
    with head_col:
        st.markdown('<h2 style="font-size: 1.6rem; margin: 0 0 4px 0;">Dokumen Resmi</h2>', unsafe_allow_html=True)
        st.caption("Daftar dokumen operasional terdaftar dalam sistem PT Majelis Debu Bintang:")
    with action_col:
        if st.button("+ Dokumen Baru", type="primary", use_container_width=True, key="btn_open_modal_doc"):
            modal_tambah_dokumen()

    st.markdown('<div style="margin-top: 1.25rem;"></div>', unsafe_allow_html=True)

    dept_groups = {}
    for d in docs:
        dept_groups.setdefault(d.department, []).append(d)

    for dept_name, dept_docs in dept_groups.items():
        st.markdown(f"**Departemen: {dept_name}**")
        for d in dept_docs:
            with st.expander(f"{d.doc_id} &mdash; {d.title} (Akses: {', '.join(d.access_roles)})"):
                st.caption(f"Versi: {d.version} &middot; Tanggal: {d.effective_date}")
                st.markdown("---")
                st.markdown(d.content)
        st.write("")


