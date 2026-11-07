"""Dataset utility to download or prepare benchmark datasets (FiQA, Wikipedia ID)."""

import argparse
import json
from pathlib import Path

FIQA_DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def prepare_synthetic_wikipedia_id(output_path: Path, num_articles: int = 20):
    """Generates synthetic Indonesian Wikipedia-style public articles for domain expansion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    articles = [
        {
            "doc_id": "wiki_id_001",
            "title": "Kecerdasan Buatan di Indonesia",
            "text": "Kecerdasan buatan (AI) di Indonesia berkembang pesat dengan strategi nasional kecerdasan buatan (Stranas KA) yang dicanangkan oleh BPPT dan Kementerian Kominfo. Penerapan utama mencakup layanan kesehatan, reformasi birokrasi, pendidikan dan riset, ketahanan pangan, serta kota cerdas.",
            "access_roles": ["public"],
            "department": "Public Knowledge",
        },
        {
            "doc_id": "wiki_id_002",
            "title": "Sejarah dan Kedaulatan Data Nasional",
            "text": "Undang-Undang Perlindungan Data Pribadi (UU PDP) Nomor 27 Tahun 2022 mengatur hak pemilik data pribadi dan kewajiban pengendali data korporasi di Indonesia. Korporasi diwajibkan menerapkan standar tata kelola dan kontrol akses yang ketat untuk mencegah kebocoran informasi rahasia.",
            "access_roles": ["public"],
            "department": "Public Knowledge",
        },
        {
            "doc_id": "wiki_id_003",
            "title": "Teknologi Big Data dan Apache Hadoop",
            "text": "Teknologi Big Data di industri telekomunikasi dan perbankan Indonesia banyak mengadopsi ekosistem Apache Hadoop, Apache Spark, dan Elasticsearch untuk pemrosesan transaksi berkapasitas petabyte secara terdistribusi dan reliabel.",
            "access_roles": ["public"],
            "department": "Public Knowledge",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    print(f"Generated {len(articles)} Indonesian Wikipedia benchmark articles to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download / Prepare Benchmark Datasets")
    parser.add_argument("--dataset", choices=["wiki_id", "fiqa"], default="wiki_id")
    parser.add_argument("--output", default="data/raw/wiki_id/corpus_wiki_id.jsonl")
    args = parser.parse_args()

    if args.dataset == "wiki_id":
        prepare_synthetic_wikipedia_id(Path(args.output))
    else:
        print(f"BEIR FiQA sample is already available in: data/raw/fiqa_sample/")


if __name__ == "__main__":
    main()
