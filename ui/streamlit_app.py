"""Streamlit Interactive Web UI for Majelis Debu Bintang.

Minimalist, clean, anti-slop interface inspired by Claude & Linear.
Strictly zero clutter, concise role labels, quiet micro-interactions,
and emojis reserved strictly for user (👤) and assistant (🤖) avatars.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
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
    page_title="Majelis Debu Bintang",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist Custom CSS - Clean, Quiet, Anti-Slop
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #1A1917;
        --surface: #242321;
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
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 860px !important;
    }

    /* Clean Pill Tabs (Zero red lines, zero clutter) */
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
    div[data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9999px !important;
        padding: 3px !important;
        gap: 2px !important;
        margin-bottom: 2rem !important;
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

    /* Floating Chat Input Bar */
    div[data-testid="stChatInput"] {
        background: transparent !important;
        padding-top: 1rem !important;
    }
    div[data-testid="stChatInput"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        padding: 4px 8px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 2px rgba(217, 119, 87, 0.2) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.92rem !important;
        color: var(--text) !important;
    }
    div[data-testid="stChatInput"] button {
        background: var(--terracotta) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 30px !important;
        height: 30px !important;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.75rem 0 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #232220 !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 12px 18px !important;
        margin-bottom: 0.75rem !important;
    }
    div[data-testid="stChatMessage"] p {
        line-height: 1.65 !important;
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

    /* Refusal Callout */
    .abstain-box {
        background: var(--surface);
        border-left: 3px solid var(--terracotta);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: var(--text);
        margin: 8px 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #151413 !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem !important;
    }

    /* Minimalist Radio Selector */
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

    /* Clean Action Button */
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

# --- SIDEBAR: Access Control (Clean & Uncluttered) ---
with st.sidebar:
    st.markdown('<div class="serif-brand" style="font-size: 1.35rem; margin-bottom: 2px;">Majelis Debu Bintang</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.8rem; color: #8E8980; margin-bottom: 1.5rem;">Asisten Dokumen & Pengetahuan</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.78rem; font-weight: 600; color: #FAF7F2; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Peran Aktif</div>', unsafe_allow_html=True)

    # Clean, concise role names without verbose parenthetical explanations
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
            <h1 style="font-size: 2.1rem; margin: 0 0 6px 0;">Majelis Debu Bintang</h1>
            <div style="color: #8E8980; font-size: 0.95rem;">
                Asisten dokumen resmi berbasis peran (RBAC) dengan sitasi terverifikasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sleek Horizontal Prompt Pills (No duplicate boxes, no visual clutter)
    p1, p2, p3, p4 = st.columns(4)
    quick_query = None

    if p1.button("Sandi Laptop IT"):
        quick_query = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"
    if p2.button("Hak Cuti Tahunan"):
        quick_query = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"
    if p3.button("Uji ACL (Gaji)"):
        quick_query = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"
    if p4.button("Uji Abstention"):
        quick_query = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"

    # Chat Input Box
    query_input = st.chat_input("Tanyakan sesuatu tentang dokumen organisasi...")
    active_query = quick_query or query_input

    if active_query:
        user = UserContext(user_id="user_active", roles=[selected_role])

        # Cute user avatar only
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**[{role_options[selected_role]}]** &mdash; {active_query}")

        t0 = time.perf_counter()

        # Step 1: Hybrid Retrieval with ACL pre-filtering
        fused = retriever.search(active_query, user, top_k=settings.top_k_fused)

        # Step 2: Cross-Encoder Reranking
        reranked = reranker.rerank(active_query, fused, top_k=settings.top_k_final)

        # Step 3: Confidence Gating
        should_abstain, refusal_reason, conf_score = gate.evaluate(reranked)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Cute assistant avatar only
        with st.chat_message("assistant", avatar="🤖"):
            if should_abstain:
                st.markdown(
                    f"""
                    <div class="abstain-box">
                        {refusal_reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Latensi: {latency_ms:.1f} ms &middot; Skor: {conf_score:.3f} &middot; Status: Ditolak")
            else:
                answer = generator.generate(active_query, reranked, [selected_role])
                citations, precision, _ = verifier.extract_and_verify(answer, reranked)

                st.markdown(answer)

                if citations:
                    citation_html = '<div style="margin-top: 8px;">'
                    for cit in citations:
                        citation_html += f'<span class="citation-tag">{cit.doc_id} &middot; {cit.section} (Hal. {cit.page})</span>'
                    citation_html += '</div>'
                    st.markdown(citation_html, unsafe_allow_html=True)

                st.caption(f"Latensi: {latency_ms:.1f} ms &middot; Keyakinan: {conf_score:.3f} &middot; Peran: {role_options[selected_role]}")

            # Diagnostics (Quiet Expander)
            with st.expander("Detail Retrieval & Diagnostik"):
                st.markdown(f"**Status Gerbang:** `{'Ditolak' if should_abstain else 'Lolos'}`")
                if reranked:
                    diag_data = []
                    for c in reranked:
                        diag_data.append({
                            "Rank": c.rank,
                            "Doc ID": c.chunk.doc_id,
                            "Bagian": c.chunk.section,
                            "Akses": ", ".join(c.chunk.access_roles),
                            "BM25": f"{c.lexical_score:.2f}",
                            "Rerank": f"{c.rerank_score:.3f}" if c.rerank_score else "-",
                            "Cuplikan": c.chunk.text[:90] + "...",
                        })
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                else:
                    st.caption("Tidak ada dokumen yang lolos filter hak akses pengguna.")

with tab_eval:
    st.markdown('<h2 style="font-size: 1.6rem; margin-bottom: 2px;">Evaluasi & Benchmark</h2>', unsafe_allow_html=True)
    st.caption("Uji kuantitatif Information Retrieval, Keamanan Hak Akses (ACL), dan Abstention.")

    if st.button("Jalankan Evaluasi"):
        with st.spinner("Menjalankan evaluasi..."):
            harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
            results = harness.run_ablation()
            EvaluationReporter.save_report(
                rows=results,
                output_md_path=Path("reports/ablation_results.md"),
                output_csv_path=Path("reports/ablation_results.csv"),
            )
            st.success("Evaluasi selesai.")

    # Read existing report
    report_file = Path("reports/ablation_results.md")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.caption("Laporan belum tersedia. Klik tombol di atas untuk menjalankan.")

with tab_corpus:
    st.markdown('<h2 style="font-size: 1.6rem; margin-bottom: 2px;">Dokumen Resmi</h2>', unsafe_allow_html=True)
    st.caption("Daftar dokumen operasional terdaftar dalam sistem Majelis Debu Bintang:")

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
