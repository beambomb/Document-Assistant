"""Streamlit Interactive Web UI for Enterprise Document Assistant.

Designed with clean typography, restrained enterprise aesthetic,
and zero irrelevant emojis (emojis reserved exclusively for user & assistant avatars).
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

# Page configuration - clean title, no decorative emojis
st.set_page_config(
    page_title="Enterprise Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for high-signal enterprise design
st.markdown(
    """
    <style>
    /* Global Typography & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Clean, quiet headers */
    h1, h2, h3 {
        letter-spacing: -0.02em;
        font-weight: 600;
    }
    
    /* Subtle borders and clean cards */
    .stAlert {
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Quick query pill buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px;
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: transparent;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: rgba(255, 255, 255, 0.35);
        background: rgba(255, 255, 255, 0.04);
    }
    
    /* Citation badges */
    .citation-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        border-radius: 4px;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        font-family: monospace;
    }
    
    /* Metadata footer in sidebar */
    .sidebar-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    
    /* Quiet divider */
    hr {
        margin: 1.25rem 0;
        border-color: rgba(255, 255, 255, 0.08);
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

# --- SIDEBAR: Access Control (RBAC) & System State ---
st.sidebar.subheader("Kontrol Akses (RBAC)")
st.sidebar.caption("Pilih peran aktif untuk menguji batas akses dokumen:")

role_labels = {
    "guest": "Tamu (Hanya dokumen publik)",
    "employee": "Karyawan (SOP IT, Cuti, Umum)",
    "finance": "Keuangan (Klaim & Reimbursement)",
    "hr": "HR & Personalia (Kebijakan Cuti & Disipliner)",
    "management": "Direksi / Manajemen (Akses Penuh)",
    "admin": "Administrator Sistem",
}

selected_role = st.sidebar.radio(
    "Peran Aktif Pengguna",
    options=list(role_labels.keys()),
    format_func=lambda r: role_labels[r],
    index=1,
    label_visibility="collapsed",
)

st.sidebar.divider()

# Clean metadata display
active_provider = settings.llm_provider.lower()
if active_provider in ["gguf", "local"]:
    model_name = Path(settings.local_model_path).name
else:
    model_name = settings.openrouter_model if active_provider == "openrouter" else "Mock"

st.sidebar.markdown(
    f"""
    <div class="sidebar-meta">
        <div><strong>Dokumen Terindeks:</strong> {len(docs)} SOP ({len(chunks)} Chunks)</div>
        <div><strong>Penyedia LLM:</strong> {active_provider.upper()}</div>
        <div><strong>Model:</strong> {model_name}</div>
        <div><strong>Threshold Gate:</strong> {settings.confidence_threshold}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- MAIN INTERFACE TABS ---
tab_chat, tab_eval, tab_corpus = st.tabs([
    "Asisten Dokumen",
    "Evaluasi & Benchmark",
    "Daftar Dokumen",
])

with tab_chat:
    st.title("Enterprise Document Assistant")
    st.caption("Pencarian dokumen korporat berbasis peran (RBAC) dengan sitasi wajib dan pencegahan halusinasi.")

    # Quick query test pills
    st.caption("Pertanyaan Uji Cepat:")
    col1, col2, col3, col4 = st.columns(4)
    quick_query = None
    if col1.button("Kebijakan Sandi IT"):
        quick_query = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"
    if col2.button("Hak Cuti Tahunan"):
        quick_query = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"
    if col3.button("Uji ACL: Remunerasi Direksi"):
        quick_query = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"
    if col4.button("Uji Jebakan: Resep Makanan"):
        quick_query = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"

    query_input = st.chat_input("Ketik pertanyaan mengenai dokumen resmi perusahaan...")
    active_query = quick_query or query_input

    if active_query:
        user = UserContext(user_id="demo_user", roles=[selected_role])

        # Cute user avatar only
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**[{selected_role.capitalize()}]**: {active_query}")

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
                st.warning(refusal_reason)
                st.caption(
                    f"Latensi: {latency_ms:.1f} ms | Skor Relevansi: {conf_score:.3f} | Status: Ditolak (Abstain)"
                )
            else:
                answer = generator.generate(active_query, reranked, [selected_role])
                citations, precision, _ = verifier.extract_and_verify(answer, reranked)

                st.markdown(answer)

                if citations:
                    st.markdown("---")
                    st.caption("Sitasi Terverifikasi:")
                    citation_html = ""
                    for cit in citations:
                        citation_html += (
                            f'<span class="citation-badge">'
                            f'Doc: {cit.doc_id} &middot; {cit.section} (Hal. {cit.page})'
                            f'</span> '
                        )
                    st.markdown(citation_html, unsafe_allow_html=True)

                st.caption(
                    f"Latensi: {latency_ms:.1f} ms | Skor Keyakinan: {conf_score:.3f} | Role: {selected_role.capitalize()}"
                )

            # Diagnostics Accordion
            with st.expander("Detail Retrieval & Diagnostik"):
                st.markdown(
                    f"**Status Gerbang Keyakinan:** `{'Ditolak (Abstain)' if should_abstain else 'Lolos (Grounded)'}`"
                )
                if reranked:
                    diag_data = []
                    for c in reranked:
                        diag_data.append({
                            "Rank": c.rank,
                            "Doc ID": c.chunk.doc_id,
                            "Bagian": c.chunk.section,
                            "Hak Akses": ", ".join(c.chunk.access_roles),
                            "Skor BM25": f"{c.lexical_score:.2f}",
                            "Skor Rerank": f"{c.rerank_score:.3f}" if c.rerank_score else "-",
                            "Cuplikan Teks": c.chunk.text[:90] + "...",
                        })
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                else:
                    st.caption("Tidak ada dokumen yang lolos filter hak akses pengguna.")

with tab_eval:
    st.subheader("Evaluasi & Benchmark Model")
    st.caption("Pengujian Information Retrieval (IR), Keamanan Hak Akses (ACL), dan Kemampuan Penolakan (Abstention):")

    if st.button("Jalankan Evaluasi Ulang"):
        with st.spinner("Menjalankan harness evaluasi..."):
            harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
            results = harness.run_ablation()
            # Save report
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
        st.caption("Laporan belum tersedia. Klik tombol di atas untuk menjalankan evaluasi.")

with tab_corpus:
    st.subheader("Direktori Dokumen Resmi")
    st.caption("Daftar dokumen operasional yang terdaftar dalam indeks sistem:")

    # Group documents by department
    dept_groups = {}
    for d in docs:
        dept_groups.setdefault(d.department, []).append(d)

    for dept_name, dept_docs in dept_groups.items():
        st.markdown(f"**Departemen: {dept_name}**")
        for d in dept_docs:
            with st.expander(f"{d.doc_id} — {d.title} (Akses: {', '.join(d.access_roles)})"):
                st.caption(f"Versi: {d.version} | Departemen: {d.department} | Berlaku: {d.effective_date}")
                st.markdown("---")
                st.markdown(d.content)
        st.write("")
