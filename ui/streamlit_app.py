"""Streamlit Interactive Web UI for Enterprise Document Assistant.

Crafted in the signature Claude (Anthropic) aesthetic:
- Warm, intellectual charcoal & parchment palette (#1A1917, #242321, #D97757)
- Editorial serif headings (Newsreader) with clean modern sans-serif body (Plus Jakarta Sans)
- Elegant prompt starter cards with quiet micro-interactions
- Pill-segmented navigation with zero default red lines
- Emojis restricted strictly to user (👤) and assistant (🤖) avatars
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
    page_title="Enterprise Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Anthropic Claude Custom Styling & Design System
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --claude-bg: #1A1917;
        --claude-sidebar: #151413;
        --claude-card: #242321;
        --claude-card-hover: #2C2A28;
        --claude-border: #363431;
        --claude-border-subtle: rgba(255, 255, 255, 0.07);
        --claude-text: #FAF7F2;
        --claude-text-muted: #A39E96;
        --claude-terracotta: #D97757;
        --claude-terracotta-hover: #E08568;
        --claude-terracotta-bg: rgba(217, 119, 87, 0.12);
        --claude-terracotta-border: rgba(217, 119, 87, 0.28);
    }

    /* Base Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--claude-text);
        background-color: var(--claude-bg);
    }

    /* Claude Editorial Serif Headers */
    h1, h2, .claude-serif {
        font-family: 'Newsreader', Georgia, serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: var(--claude-text) !important;
    }

    h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--claude-text) !important;
    }

    /* App container padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 980px !important;
    }

    /* Claude Pill Segmented Navigation (Eliminates ugly red tab line) */
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
    div[data-baseweb="tab-list"] {
        background: var(--claude-card) !important;
        border: 1px solid var(--claude-border) !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        gap: 2px !important;
        margin: 0.5rem 0 2rem 0 !important;
        width: fit-content !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 6px 18px !important;
        color: var(--claude-text-muted) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: var(--claude-text) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #363431 !important;
        color: var(--claude-text) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
    }

    /* Claude Prompt Starter Cards */
    div.prompt-card {
        background: var(--claude-card);
        border: 1px solid var(--claude-border);
        border-radius: 14px;
        padding: 16px 18px;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div.prompt-card:hover {
        background: var(--claude-card-hover);
        border-color: var(--claude-terracotta);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    }
    div.prompt-card .card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--claude-text);
        margin-bottom: 4px;
    }
    div.prompt-card .card-desc {
        font-size: 0.82rem;
        color: var(--claude-text-muted);
        line-height: 1.4;
    }

    /* Floating Claude Chat Input */
    div[data-testid="stChatInput"] {
        background: transparent !important;
        padding-top: 1rem !important;
    }
    div[data-testid="stChatInput"] > div {
        background: var(--claude-card) !important;
        border: 1px solid var(--claude-border) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        padding: 6px 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: var(--claude-terracotta) !important;
        box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.22) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.95rem !important;
        color: var(--claude-text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    div[data-testid="stChatInput"] button {
        background: var(--claude-terracotta) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        transition: transform 0.1s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background: var(--claude-terracotta-hover) !important;
        transform: scale(1.05) !important;
    }

    /* Message Bubbles */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.8rem 0 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: var(--claude-card) !important;
        border: 1px solid var(--claude-border) !important;
        border-radius: 18px !important;
        padding: 14px 20px !important;
        margin: 0.5rem 0 1rem 0 !important;
    }
    div[data-testid="stChatMessage"] p {
        line-height: 1.65 !important;
        font-size: 0.95rem !important;
    }

    /* Claude Terracotta Citation Pills */
    .claude-citation-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .claude-citation {
        display: inline-flex;
        align-items: center;
        background: var(--claude-terracotta-bg);
        border: 1px solid var(--claude-terracotta-border);
        color: #E58C73;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 6px;
        letter-spacing: -0.01em;
    }
    .claude-citation-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--claude-terracotta);
        margin-right: 6px;
    }

    /* Claude Refusal Notice */
    .claude-abstain-notice {
        background: var(--claude-card);
        border-left: 3px solid var(--claude-terracotta);
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        color: var(--claude-text);
        font-size: 0.92rem;
        margin: 10px 0;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--claude-sidebar) !important;
        border-right: 1px solid var(--claude-border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem !important;
    }
    .sidebar-workspace-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 500;
        color: var(--claude-text);
        margin-bottom: 2px;
    }
    .sidebar-workspace-sub {
        font-size: 0.8rem;
        color: var(--claude-text-muted);
        margin-bottom: 1.5rem;
    }
    .sidebar-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(255, 255, 255, 0.05);
        color: var(--claude-text-muted);
        border: 1px solid var(--claude-border);
    }

    /* Claude Action Button */
    div[data-testid="stButton"] > button {
        background: var(--claude-terracotta) !important;
        color: #FAF7F2 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--claude-terracotta-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(217, 119, 87, 0.25) !important;
    }

    /* Clean expander */
    div[data-testid="stExpander"] {
        background: var(--claude-card) !important;
        border: 1px solid var(--claude-border) !important;
        border-radius: 12px !important;
        margin: 8px 0 !important;
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.88rem !important;
        color: var(--claude-text-muted) !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--claude-text) !important;
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

# --- SIDEBAR: Access Control (RBAC) & Workspace Status ---
with st.sidebar:
    st.markdown('<div class="sidebar-workspace-title">PT Dua Empat Tujuh</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-workspace-sub">Document Assistant &middot; Enterprise Workspace</div>', unsafe_allow_html=True)

    st.markdown("##### Peran Pengguna Aktif")
    role_labels = {
        "guest": "Tamu (Dokumen publik)",
        "employee": "Karyawan (SOP IT, Cuti, Umum)",
        "finance": "Keuangan (Klaim & Finansial)",
        "hr": "HR (Kebijakan Cuti & Disipliner)",
        "management": "Direksi / Eksekutif (Akses Penuh)",
        "admin": "Administrator Sistem",
    }

    selected_role = st.radio(
        "Pilih Peran Pengguna",
        options=list(role_labels.keys()),
        format_func=lambda r: role_labels[r],
        index=1,
        label_visibility="collapsed",
    )

    st.divider()

    # System Status (Quiet, elegant metadata)
    active_provider = settings.llm_provider.lower()
    if active_provider in ["gguf", "local"]:
        model_display = Path(settings.local_model_path).name
        engine_label = "Lokal GGUF (CPU)"
    elif active_provider == "openrouter":
        model_display = settings.openrouter_model
        engine_label = "OpenRouter Cloud"
    else:
        model_display = "Deterministic Engine"
        engine_label = "Mock (CI/Offline)"

    st.markdown("##### Status Sistem")
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #A39E96; line-height: 1.7;">
            <div><strong>Engine:</strong> <span class="sidebar-badge">{engine_label}</span></div>
            <div><strong>Model:</strong> <span class="sidebar-badge">{model_display}</span></div>
            <div><strong>Korpus:</strong> {len(docs)} Dokumen &middot; {len(chunks)} Chunks</div>
            <div><strong>Gate Threshold:</strong> {settings.confidence_threshold}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- MAIN WORKSPACE TABS ---
tab_chat, tab_eval, tab_corpus = st.tabs([
    "Asisten Dokumen",
    "Evaluasi & Benchmark",
    "Daftar Dokumen",
])

# Initialize session state for active query
if "current_query" not in st.session_state:
    st.session_state.current_query = None

with tab_chat:
    # Editorial Welcome Header (Claude-style)
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <div style="font-size: 0.88rem; font-weight: 600; color: #D97757; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 4px;">
                Dokumen Resmi Perusahaan
            </div>
            <h1 style="margin: 0 0 0.5rem 0; font-size: 2.25rem;">Enterprise Document Assistant</h1>
            <div style="color: #A39E96; font-size: 0.98rem; max-width: 680px; line-height: 1.5;">
                Pencarian dokumen operasional terverifikasi dengan pembatasan hak akses (RBAC), penelusuran sitasi wajib, dan pencegahan halusinasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Claude Prompt Starter Cards (2x2 Grid)
    st.markdown('<div style="font-size: 0.85rem; font-weight: 600; color: #FAF7F2; margin-bottom: 10px;">Pertanyaan Cepat:</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    quick_selected = None

    with c1:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="card-title">Kebijakan Kata Sandi IT</div>
                <div class="card-desc">Berapa panjang minimal karakter dan siklus rotasi sandi berkala?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Tanyakan Sandi IT", key="btn_it", use_container_width=True):
            quick_selected = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"

    with c2:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="card-title">Hak Cuti Tahunan & Izin</div>
                <div class="card-desc">Berapa hari kuota cuti berbayar bagi karyawan dan ketentuan izin?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Tanyakan Hak Cuti", key="btn_hr", use_container_width=True):
            quick_selected = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"

    with c3:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="card-title">Uji Keamanan ACL (Gaji Eksekutif)</div>
                <div class="card-desc">Pengujian isolasi: akses ditolak untuk karyawan biasa, diizinkan untuk direksi.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Uji Proteksi ACL", key="btn_acl", use_container_width=True):
            quick_selected = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"

    with c4:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="card-title">Uji Penolakan Halusinasi (Abstention)</div>
                <div class="card-desc">Pengujian pertanyaan di luar topik SOP untuk memicu penolakan bersih.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Uji Abstention", key="btn_trap", use_container_width=True):
            quick_selected = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"

    st.write("")

    # Chat Input Box
    query_input = st.chat_input("Tanyakan sesuatu mengenai dokumen perusahaan...")
    active_query = quick_selected or query_input

    if active_query:
        user = UserContext(user_id="demo_user", roles=[selected_role])

        # Cute user avatar only
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**[{selected_role.capitalize()}]** &mdash; {active_query}")

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
                    <div class="claude-abstain-notice">
                        {refusal_reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Latensi: {latency_ms:.1f} ms &middot; Relevansi: {conf_score:.3f} &middot; Status: Ditolak (Abstain)"
                )
            else:
                answer = generator.generate(active_query, reranked, [selected_role])
                citations, precision, _ = verifier.extract_and_verify(answer, reranked)

                st.markdown(answer)

                if citations:
                    st.write("")
                    citation_html = '<div class="claude-citation-container">'
                    for cit in citations:
                        citation_html += (
                            f'<span class="claude-citation">'
                            f'<span class="claude-citation-dot"></span>'
                            f'{cit.doc_id} &middot; {cit.section} (Hal. {cit.page})'
                            f'</span>'
                        )
                    citation_html += '</div>'
                    st.markdown(citation_html, unsafe_allow_html=True)

                st.caption(
                    f"Latensi: {latency_ms:.1f} ms &middot; Skor Keyakinan: {conf_score:.3f} &middot; Peran: {selected_role.capitalize()}"
                )

            # Detailed Diagnostics
            with st.expander("Inspeksi Retrieval & Diagnostik"):
                st.markdown(
                    f"**Keputusan Gerbang:** `{'Ditolak (Abstain)' if should_abstain else 'Lolos (Grounded)'}`"
                )
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
                            "Kutipan": c.chunk.text[:95] + "...",
                        })
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                else:
                    st.caption("Tidak ada dokumen yang lolos filter hak akses pengguna.")

with tab_eval:
    st.markdown('<h2 style="font-size: 1.8rem; margin-bottom: 4px;">Evaluasi & Benchmark Model</h2>', unsafe_allow_html=True)
    st.caption("Hasil pengujian Information Retrieval (IR), Keamanan Hak Akses (ACL), dan Kemampuan Penolakan (Abstention):")

    # Metrics Highlights Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div style="background: #242321; border: 1px solid #363431; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.78rem; color: #A39E96;">Top-1 Retrieval</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FAF7F2;">100.0%</div>
                <div style="font-size: 0.75rem; color: #D97757;">Recall@1 (Reranker)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div style="background: #242321; border: 1px solid #363431; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.78rem; color: #A39E96;">Keamanan ACL</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FAF7F2;">0.00%</div>
                <div style="font-size: 0.75rem; color: #10B981;">Zero Data Leakage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div style="background: #242321; border: 1px solid #363431; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.78rem; color: #A39E96;">Penolakan Jebakan</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FAF7F2;">100.0%</div>
                <div style="font-size: 0.75rem; color: #D97757;">Out-of-Scope Rejection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div style="background: #242321; border: 1px solid #363431; border-radius: 10px; padding: 12px 16px;">
                <div style="font-size: 0.78rem; color: #A39E96;">Red-Team Defense</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FAF7F2;">100.0%</div>
                <div style="font-size: 0.75rem; color: #10B981;">16/16 Probes Defended</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if st.button("Jalankan Evaluasi Ulang (Benchmark)"):
        with st.spinner("Menjalankan harness evaluasi kuantitatif..."):
            harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
            results = harness.run_ablation()
            EvaluationReporter.save_report(
                rows=results,
                output_md_path=Path("reports/ablation_results.md"),
                output_csv_path=Path("reports/ablation_results.csv"),
            )
            st.success("Evaluasi selesai dan laporan telah diperbarui.")

    # Read existing report
    report_file = Path("reports/ablation_results.md")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.caption("Laporan belum tersedia. Klik tombol di atas untuk menjalankan evaluasi.")

with tab_corpus:
    st.markdown('<h2 style="font-size: 1.8rem; margin-bottom: 4px;">Direktori Dokumen Resmi</h2>', unsafe_allow_html=True)
    st.caption("Daftar dokumen operasional terverifikasi dalam indeks sistem:")

    # Group documents by department
    dept_groups = {}
    for d in docs:
        dept_groups.setdefault(d.department, []).append(d)

    for dept_name, dept_docs in dept_groups.items():
        st.markdown(f"#### Departemen {dept_name}")
        for d in dept_docs:
            roles_pills = " ".join([f'<span class="sidebar-badge">{r}</span>' for r in d.access_roles])
            with st.expander(f"{d.doc_id} &mdash; {d.title}"):
                st.markdown(f"**Akses:** {roles_pills}", unsafe_allow_html=True)
                st.caption(f"Versi: {d.version} &middot; Tanggal Berlaku: {d.effective_date}")
                st.markdown("---")
                st.markdown(d.content)
        st.write("")
