"""Streamlit Interactive Web UI with Role-Switching and Search Diagnostics."""

import time
import streamlit as st
import pandas as pd
from pathlib import Path

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

st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="🏢",
    layout="wide",
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

# --- SIDEBAR: Role Switcher & System Status ---
st.sidebar.title("🔐 Kontrol Akses (RBAC)")
st.sidebar.markdown("Pilih peran pengguna untuk menguji isolasi hak akses dokumen:")

role_options = {
    "guest": "👤 Tamu / Publik (Hanya Profil)",
    "employee": "👔 Karyawan (SOP IT, Cuti, Umum)",
    "finance": "💳 Keuangan (SOP Reimbursement & Klaim)",
    "hr": "📋 HR & Personalia (Kebijakan Cuti & Disipliner)",
    "management": "👑 Direksi / Manajemen (Akses Penuh & Gaji Direksi)",
    "admin": "⚙️ Administrator Sistem",
}

selected_role = st.sidebar.radio(
    "Peran Aktif Pengguna Saat Ini:",
    options=list(role_options.keys()),
    format_func=lambda x: role_options[x],
    index=1,  # Default to employee
)

st.sidebar.divider()
st.sidebar.markdown(f"**Total Dokumen**: `{len(docs)} SOP`")
st.sidebar.markdown(f"**Total Chunk Terindeks**: `{len(chunks)} Chunks`")
st.sidebar.markdown(f"**LLM Provider**: `{settings.llm_provider.upper()}`")
st.sidebar.markdown(f"**Model**: `{settings.openrouter_model if settings.llm_provider=='openrouter' else 'Mock Deterministic'}`")

# --- MAIN TABS ---
tab_chat, tab_eval, tab_corpus = st.tabs(["💬 Asisten Dokumen", "📊 Dashboard Evaluasi & Ablasi", "📁 Daftar Dokumen"])

with tab_chat:
    st.title("🏢 Enterprise Document Assistant")
    st.caption("RAG Berstandar Korporasi: Kontrol Akses per Pengguna, Sitasi Wajib, & Gerbang Penolakan Halusinasi.")

    # Quick Example Buttons
    st.markdown("**Uji Cepat Pertanyaan:**")
    col1, col2, col3, col4 = st.columns(4)
    quick_query = None
    if col1.button("🔑 Password Laptop (IT)"):
        quick_query = "Berapa panjang minimal kata sandi laptop dan seberapa sering wajib diperbarui?"
    if col2.button("🏖️ Hak Cuti Karyawan (HR)"):
        quick_query = "Berapa hari hak cuti tahunan dan bagaimana aturan cuti melahirkan?"
    if col3.button("🚨 Gaji Direksi (Uji Rahasia/ACL)"):
        quick_query = "Berapa nominal gaji pokok bulanan dan bonus tahunan CEO dan Direksi?"
    if col4.button("🍲 Resep Rendang (Uji Jebakan)"):
        quick_query = "Bagaimana resep dan bumbu halus memasak rendang daging sapi Padang?"

    query_input = st.chat_input("Ketik pertanyaan Anda tentang dokumen perusahaan...")
    active_query = quick_query or query_input

    if active_query:
        user = UserContext(user_id="demo_user", roles=[selected_role])

        with st.chat_message("user"):
            st.markdown(f"**[{selected_role.upper()}]**: {active_query}")

        t0 = time.perf_counter()

        # Step 1: Hybrid Search with ACL pre-filtering
        fused = retriever.search(active_query, user, top_k=settings.top_k_fused)

        # Step 2: Cross-Encoder Reranking
        reranked = reranker.rerank(active_query, fused, top_k=settings.top_k_final)

        # Step 3: Confidence Gating
        should_abstain, refusal_reason, conf_score = gate.evaluate(reranked)

        latency_ms = (time.perf_counter() - t0) * 1000

        with st.chat_message("assistant"):
            if should_abstain:
                st.warning(f"🚫 **PERMINTAAN DITOLAK (ABSTAIN):**\n\n{refusal_reason}")
                st.info("💡 *Alasan Keamanan/Akurasi:* Dokumen tidak ditemukan dalam cakupan hak akses role Anda atau pertanyaan di luar topik dokumen perusahaan.")
            else:
                answer = generator.generate(active_query, reranked, [selected_role])
                citations, precision, _ = verifier.extract_and_verify(answer, reranked)

                st.markdown(answer)

                if citations:
                    st.markdown("---")
                    st.markdown("**📌 Sitasi Dokumen Terverifikasi:**")
                    for cit in citations:
                        st.markdown(f"- 📄 **[{cit.doc_id}]** *{cit.title}* - Bagian: `{cit.section}` (Hal. {cit.page})")

            st.caption(f"⏱️ Latensi: `{latency_ms:.1f} ms` | Skor Keyakinan: `{conf_score:.3f}` | Role: `{selected_role}`")

            # Diagnostics Accordion
            with st.expander("🔍 Detail Inspeksi Retrieval & Diagnostik"):
                st.markdown(f"**Status Gerbang Keyakinan (Confidence Gate)**: `{'Ditolak (Abstain)' if should_abstain else 'Lolos (Grounded)'}`")
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
                            "Cuplikan": c.chunk.text[:80] + "...",
                        })
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
                else:
                    st.write("Tidak ada dokumen yang lolos filter hak akses pengguna.")

with tab_eval:
    st.header("📊 Laporan Hasil Evaluasi & Ablasi Model")
    st.markdown("Hasil pengujian kuantitatif Information Retrieval (IR), Keamanan ACL, dan Kemampuan Abstention:")

    if st.button("🚀 Jalankan Evaluasi Ulang (Benchmark)"):
        with st.spinner("Menjalankan harness evaluasi..."):
            harness = EvaluationHarness(dataset_name="id_enterprise", use_mock_embeddings=True)
            results = harness.run_ablation()
            st.success("Evaluasi selesai!")
            st.markdown(EvaluationReporter.generate_markdown_table(results))

    # Read existing report if exists
    report_file = Path("reports/ablation_results.md")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Laporan evaluasi belum di-generate. Klik tombol di atas untuk menjalankan.")

with tab_corpus:
    st.header("📁 Daftar Dokumen Korpus Perusahaan")
    for d in docs:
        with st.expander(f"📄 [{d.doc_id}] {d.title} (Akses: {', '.join(d.access_roles)})"):
            st.markdown(f"**Versi**: {d.version} | **Divisi**: {d.department} | **Tanggal Berlaku**: {d.effective_date}")
            st.markdown("---")
            st.markdown(d.content)
