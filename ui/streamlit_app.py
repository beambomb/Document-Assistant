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

    /* Container width & vertical breathing room */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 860px !important;
    }

    /* Clean Pill Tabs (Zero red lines) */
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
    div[data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        padding: 3px !important;
        gap: 2px !important;
        margin-bottom: 1.5rem !important;
        width: fit-content !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 6px 16px !important;
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: var(--text) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #363431 !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    /* Minimal Prompt Suggestion Pills */
    div.stHorizontalBlock div[data-testid="stButton"] button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 5px 14px !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
    }
    div.stHorizontalBlock div[data-testid="stButton"] button:hover {
        border-color: var(--terracotta) !important;
        color: var(--text) !important;
        transform: translateY(-1px) !important;
    }

    /* Conversational Thread Layout */
    .chat-thread {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* User Message - Aligned to the RIGHT */
    .msg-row-user {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 10px;
        margin-left: 15%;
    }
    .msg-bubble-user {
        background: var(--surface-user);
        border: 1px solid var(--border);
        border-radius: 18px 4px 18px 18px;
        padding: 12px 18px;
        color: var(--text);
        font-size: 0.94rem;
        line-height: 1.55;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .msg-avatar-user {
        font-size: 1.25rem;
        line-height: 1;
        padding-top: 2px;
    }

    /* Assistant Message - Aligned to the LEFT */
    .msg-row-assistant {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
        margin-right: 5%;
    }
    .msg-avatar-assistant {
        font-size: 1.35rem;
        line-height: 1;
        padding-top: 2px;
    }
    .msg-bubble-assistant {
        flex: 1;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* Citations */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(217, 119, 87, 0.1);
        border: 1px solid rgba(217, 119, 87, 0.25);
        color: #E28668;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 5px;
        margin-right: 6px;
        margin-top: 6px;
    }

    /* Refusal Callout Box */
    .abstain-box {
        background: var(--surface);
        border-left: 3px solid var(--terracotta);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
        color: var(--text);
        margin: 6px 0;
    }

    /* Bottom Docked Chat Input (Streamlit native stBottom container) */
    div[data-testid="stBottom"] {
        background-color: var(--bg) !important;
        border-top: 1px solid var(--border-subtle) !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }
    div[data-testid="stBottom"] > div {
        max-width: 860px !important;
        margin: 0 auto !important;
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

    /* Expander */
    div[data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        margin: 6px 0 !important;
    }

    /* Action Buttons */
    div[data-testid="stButton"] > button {
        background: var(--terracotta) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--terracotta-hover) !important;
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

    # Quiet System Metadata
    active_provider = settings.llm_provider.lower()
    if active_provider in ["gguf", "local"]:
        model_display = Path(settings.local_model_path).name
        engine_display = "Qwen 2.5 (Lokal CPU)"
    elif active_provider == "openrouter":
        model_display = settings.openrouter_model
        engine_display = "OpenRouter"
    else:
        model_display = "Deterministic"
        engine_display = "Mock"

    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: #8E8980; line-height: 1.7;">
            <div><strong>Engine:</strong> {engine_display}</div>
            <div><strong>Model:</strong> {model_display}</div>
            <div><strong>Indeks:</strong> {len(docs)} Dokumen &middot; {len(chunks)} Chunks</div>
            <div><strong>Gate:</strong> {settings.confidence_threshold}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- MAIN WORKSPACE TABS ---
quick_query = None
tab_chat, tab_eval, tab_corpus = st.tabs([
    "Percakapan",
    "Benchmark",
    "Dokumen",
])

with tab_chat:
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
    p1, p2, p3, p4 = st.columns(4)

    if p1.button("Sandi Laptop IT"):
        quick_query = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"
    if p2.button("Hak Cuti Tahunan"):
        quick_query = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"
    if p3.button("Uji ACL (Gaji)"):
        quick_query = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"
    if p4.button("Uji Abstention"):
        quick_query = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"

    # --- RENDER PERSISTENT CHAT HISTORY (User on RIGHT, Assistant on LEFT) ---
    if st.session_state.chat_history:
        st.markdown('<div class="chat-thread">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                # User message: aligned to the RIGHT with 👤 avatar
                clean_text = html.escape(msg["content"])
                st.markdown(
                    f"""
                    <div class="msg-row-user">
                        <div class="msg-bubble-user">
                            <div style="font-size: 0.76rem; color: #D97757; font-weight: 600; margin-bottom: 4px;">[{msg.get('role_label', 'Pengguna')}]</div>
                            {clean_text}
                        </div>
                        <div class="msg-avatar-user">👤</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Assistant message: aligned to the LEFT with 🤖 avatar
                if msg.get("abstained"):
                    refusal_escaped = html.escape(msg.get("content", ""))
                    st.markdown(
                        f"""
                        <div class="msg-row-assistant">
                            <div class="msg-avatar-assistant">🤖</div>
                            <div class="msg-bubble-assistant">
                                <div class="abstain-box">{refusal_escaped}</div>
                                <div style="font-size: 0.78rem; color: #8E8980; margin-top: 4px;">
                                    Latensi: {msg.get('latency_ms', 0):.1f} ms &middot; Skor: {msg.get('score', 0):.3f} &middot; Status: Ditolak
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    answer_text = msg.get("content", "")
                    citations = msg.get("citations", [])

                    citation_pills = ""
                    if citations:
                        citation_pills = '<div style="margin-top: 8px;">'
                        for cit in citations:
                            citation_pills += f'<span class="citation-tag">{cit["doc_id"]} &middot; {cit["section"]} (Hal. {cit["page"]})</span>'
                        citation_pills += '</div>'

                    st.markdown(
                        f"""
                        <div class="msg-row-assistant">
                            <div class="msg-avatar-assistant">🤖</div>
                            <div class="msg-bubble-assistant">
                                <div>{answer_text}</div>
                                {citation_pills}
                                <div style="font-size: 0.78rem; color: #8E8980; margin-top: 6px;">
                                    Latensi: {msg.get('latency_ms', 0):.1f} ms &middot; Keyakinan: {msg.get('score', 0):.3f}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="padding: 2.5rem 0; text-align: center; color: #8E8980; font-size: 0.9rem;">
                Belum ada percakapan. Ketik pertanyaan di bawah atau pilih pertanyaan uji cepat di atas.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_eval:
    st.markdown('<h2 style="font-size: 1.6rem; margin-bottom: 2px;">Evaluasi & Benchmark</h2>', unsafe_allow_html=True)
    st.caption("Uji kuantitatif Information Retrieval, Keamanan Hak Akses (ACL), dan Abstention.")

    if st.button("Jalankan Evaluasi Ulang"):
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

with tab_corpus:
    st.markdown('<h2 style="font-size: 1.6rem; margin-bottom: 2px;">Dokumen Resmi</h2>', unsafe_allow_html=True)
    st.caption("Daftar dokumen operasional terdaftar dalam sistem PT Majelis Debu Bintang:")

    # --- TAMBAH DOKUMEN BARU (UPLOAD & FORM INPUT) ---
    with st.expander("+ Tambah Dokumen Baru (Upload & Indeks)", expanded=False):
        st.markdown(
            '<div style="font-size: 0.88rem; color: #FAF7F2; font-weight: 500; margin-bottom: 8px;">Upload atau Buat Dokumen Baru</div>',
            unsafe_allow_html=True,
        )
        st.caption("Dokumen baru akan otomatis dipecah menjadi chunks dan diindeks secara langsung ke dalam BM25 & Dense Vector Index.")

        up_col1, up_col2 = st.columns(2)
        new_doc_id = up_col1.text_input("Nomor Dokumen", placeholder="misal: SOP-OPS-006", key="input_doc_id")
        new_title = up_col2.text_input("Judul Dokumen", placeholder="misal: SOP Pengelolaan Fasilitas Kantor", key="input_doc_title")

        up_col3, up_col4 = st.columns(2)
        dept_options = ["Umum", "Operasional", "IT", "HR", "Keuangan", "Legal"]
        new_department = up_col3.selectbox("Departemen / Divisi", options=dept_options, key="select_dept")
        new_roles = up_col4.multiselect(
            "Klasifikasi Hak Akses",
            options=["guest", "employee", "finance", "hr", "management", "admin"],
            default=["employee"],
            key="multiselect_roles",
            help="Peran pengguna mana saja yang diizinkan mengakses dokumen ini.",
        )

        uploaded_doc = st.file_uploader("Upload Dokumen (.md / .txt)", type=["md", "txt"], key="upload_doc_file")
        default_body = ""
        if uploaded_doc is not None:
            try:
                default_body = uploaded_doc.read().decode("utf-8")
            except Exception:
                default_body = ""

        new_content = st.text_area(
            "Isi / Konten Dokumen (Markdown)",
            value=default_body,
            height=160,
            placeholder="Tuliskan isi SOP, kebijakan, atau aturan operasional resmi di sini...",
            key="input_doc_content",
        )

        if st.button("Simpan & Indeks Dokumen", use_container_width=True, key="btn_save_doc"):
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

    st.write("")

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

# --- VIEWPORT-LOCKED BOTTOM CHAT INPUT (ROOT CONTAINER) ---
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
        })
    else:
        answer = generator.generate(active_query, reranked, [selected_role])
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
        })

    st.rerun()

