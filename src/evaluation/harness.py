"""Generic, reproducible Evaluation Harness for Enterprise RAG pipelines."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from src.core.models import UserContext, RetrievalCandidate, Chunk
from src.ingestion.loader import DocumentLoader
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import DenseVectorIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.confidence_gate import ConfidenceGate
from src.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    check_acl_leakage,
)
from src.evaluation.reporter import EvaluationReporter
from config.settings import settings


class EvaluationHarness:
    """Runs ablation experiments and security benchmarks across multiple domains."""

    def __init__(self, dataset_name: str = "id_enterprise", use_mock_embeddings: bool = True):
        self.dataset_name = dataset_name
        self.use_mock_embeddings = use_mock_embeddings
        self.loader = DocumentLoader()
        self.chunks: List[Chunk] = []
        self.bm25 = BM25Index()
        self.vector_idx = DenseVectorIndex(use_mock=use_mock_embeddings)
        self.reranker = CrossEncoderReranker(use_mock=True)
        self.gate = ConfidenceGate(threshold=0.35)

    def setup(self):
        """Loads corpus and builds indices."""
        if self.dataset_name == "id_enterprise":
            corpus_dir = settings.raw_data_dir / "id_enterprise"
            _, self.chunks = self.loader.load_directory_markdown(corpus_dir)
        elif self.dataset_name == "fiqa":
            corpus_file = settings.raw_data_dir / "fiqa_sample" / "corpus_fiqa_sample.jsonl"
            _, self.chunks = self.loader.load_jsonl_corpus(corpus_file)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.bm25.build(self.chunks)
        self.vector_idx.build(self.chunks)

    def load_eval_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Loads golden QA samples, trap queries, and adversarial ACL queries."""
        eval_dir = settings.eval_goldens_dir
        qa_file = eval_dir / "eval_qa_indonesian.jsonl"
        traps_file = eval_dir / "eval_traps_out_of_scope.jsonl"
        acl_file = eval_dir / "eval_acl_adversarial.jsonl"

        qa_samples, trap_samples, acl_samples = [], [], []

        if qa_file.exists():
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_samples = [json.loads(line) for line in f if line.strip()]

        if traps_file.exists():
            with open(traps_file, "r", encoding="utf-8") as f:
                trap_samples = [json.loads(line) for line in f if line.strip()]

        if acl_file.exists():
            with open(acl_file, "r", encoding="utf-8") as f:
                acl_samples = [json.loads(line) for line in f if line.strip()]

        return qa_samples, trap_samples, acl_samples

    def evaluate_mode(
        self,
        mode: str,
        qa_samples: List[Dict[str, Any]],
        trap_samples: List[Dict[str, Any]],
        acl_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluates a single retrieval configuration."""
        retriever = HybridRetriever(bm25_index=self.bm25, vector_index=self.vector_idx)

        recalls_1, recalls_3, recalls_5, mrrs, ndcgs = [], [], [], [], []
        latencies = []

        # 1. Evaluate Information Retrieval (IR) on valid QA
        for sample in qa_samples:
            query = sample["query"]
            roles = sample.get("user_roles", ["employee"])
            user = UserContext(user_id="eval_user", roles=roles)
            expected_doc_ids = sample.get("expected_doc_ids", [])

            t0 = time.perf_counter()

            if mode == "BM25 Lexical Only":
                candidates = self.bm25.search(query, user, top_k=5)
            elif mode == "Dense Vector Only":
                candidates = self.vector_idx.search(query, user, top_k=5)
            elif mode == "Hybrid (BM25 + Dense RRF)":
                candidates = retriever.search(query, user, top_k=5)
            elif mode == "Hybrid + Cross-Reranker":
                fused = retriever.search(query, user, top_k=10)
                candidates = self.reranker.rerank(query, fused, top_k=5)
            else:
                candidates = []

            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_doc_ids = [c.chunk.doc_id for c in candidates]

            recalls_1.append(recall_at_k(retrieved_doc_ids, expected_doc_ids, k=1))
            recalls_3.append(recall_at_k(retrieved_doc_ids, expected_doc_ids, k=3))
            recalls_5.append(recall_at_k(retrieved_doc_ids, expected_doc_ids, k=5))
            mrrs.append(reciprocal_rank(retrieved_doc_ids, expected_doc_ids))
            ndcgs.append(ndcg_at_k(retrieved_doc_ids, expected_doc_ids, k=5))

        # 2. Evaluate ACL Security Leakage
        leaks = 0
        for sample in acl_samples:
            query = sample["query"]
            roles = sample.get("user_roles", ["employee"])
            user = UserContext(user_id="adversary", roles=roles)
            prohibited_doc_ids = sample.get("prohibited_doc_ids", [])

            if mode == "BM25 Lexical Only":
                candidates = self.bm25.search(query, user, top_k=5)
            elif mode == "Dense Vector Only":
                candidates = self.vector_idx.search(query, user, top_k=5)
            else:
                candidates = retriever.search(query, user, top_k=5)

            retrieved_doc_ids = [c.chunk.doc_id for c in candidates]
            if check_acl_leakage(retrieved_doc_ids, prohibited_doc_ids):
                leaks += 1

        leakage_rate_pct = (leaks / len(acl_samples) * 100) if acl_samples else 0.0

        # 3. Evaluate Abstention on Out-of-Scope Traps
        correct_refusals = 0
        for sample in trap_samples:
            query = sample["query"]
            roles = sample.get("user_roles", ["employee"])
            user = UserContext(user_id="trap_user", roles=roles)

            if "Reranker" in mode:
                fused = retriever.search(query, user, top_k=10)
                candidates = self.reranker.rerank(query, fused, top_k=3)
                should_abstain, _, _ = self.gate.evaluate(candidates)
            elif "Hybrid" in mode:
                candidates = retriever.search(query, user, top_k=3)
                should_abstain = len(candidates) == 0 or candidates[0].hybrid_score < 0.025
            elif "BM25" in mode:
                candidates = self.bm25.search(query, user, top_k=3)
                should_abstain = len(candidates) == 0
            else:
                candidates = self.vector_idx.search(query, user, top_k=3)
                should_abstain = len(candidates) == 0

            if should_abstain:
                correct_refusals += 1

        trap_rejection_pct = (correct_refusals / len(trap_samples) * 100) if trap_samples else 0.0

        return {
            "mode": mode,
            "recall_at_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
            "recall_at_3": float(np.mean(recalls_3)) if recalls_3 else 0.0,
            "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "ndcg_at_5": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "leakage_rate_pct": leakage_rate_pct,
            "trap_rejection_pct": trap_rejection_pct,
            "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        }

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Runs complete ablation benchmark across all 4 modes."""
        self.setup()
        qa, traps, acl = self.load_eval_data()

        modes = [
            "BM25 Lexical Only",
            "Dense Vector Only",
            "Hybrid (BM25 + Dense RRF)",
            "Hybrid + Cross-Reranker",
        ]

        rows = []
        for mode in modes:
            result = self.evaluate_mode(mode, qa, traps, acl)
            rows.append(result)

        return rows


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Evaluation Harness")
    parser.add_argument("--dataset", default="id_enterprise", choices=["id_enterprise", "fiqa"])
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification test")
    parser.add_argument("--output-md", default="reports/ablation_results.md")
    parser.add_argument("--output-csv", default="reports/ablation_results.csv")
    args = parser.parse_args()

    harness = EvaluationHarness(dataset_name=args.dataset, use_mock_embeddings=True)
    results = harness.run_ablation()

    report_md = EvaluationReporter.save_report(
        rows=results,
        output_md_path=Path(args.output_md),
        output_csv_path=Path(args.output_csv),
    )

    print("\n" + "=" * 80)
    print("EVALUATION HARNESS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(EvaluationReporter.generate_markdown_table(results))
    print("\nSaved report to:", args.output_md)


if __name__ == "__main__":
    main()
