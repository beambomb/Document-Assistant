"""Document loader pipeline supporting Markdown corporate SOPs and JSONL benchmarks."""

import json
from pathlib import Path
from typing import List, Tuple
from src.core.models import Document, Chunk
from src.ingestion.chunker import DocumentChunker


class DocumentLoader:
    """Loads documents from raw directories and converts them into structured chunks."""

    def __init__(self, chunker: DocumentChunker | None = None):
        self.chunker = chunker or DocumentChunker()

    def load_markdown_file(self, file_path: Path) -> Tuple[Document, List[Chunk]]:
        """Loads a single markdown SOP document and chunks it."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, _ = self.chunker.parse_markdown_metadata(content)
        doc_id = metadata.get("doc_id", file_path.stem)

        doc = Document(
            doc_id=doc_id,
            title=metadata.get("title", file_path.stem),
            content=content,
            access_roles=metadata.get("access_roles", ["internal"]),
            version=metadata.get("version", "1.0"),
            effective_date=metadata.get("effective_date", "2026-01-01"),
            department=metadata.get("department", "General"),
        )

        chunks = self.chunker.chunk_document(doc)
        return doc, chunks

    def load_directory_markdown(self, dir_path: Path) -> Tuple[List[Document], List[Chunk]]:
        """Loads and chunks all markdown files within a directory."""
        all_docs: List[Document] = []
        all_chunks: List[Chunk] = []

        for md_file in sorted(dir_path.glob("*.md")):
            doc, chunks = self.load_markdown_file(md_file)
            all_docs.append(doc)
            all_chunks.extend(chunks)

        return all_docs, all_chunks

    def load_jsonl_corpus(self, file_path: Path) -> Tuple[List[Document], List[Chunk]]:
        """Loads a JSONL corpus file (e.g. FiQA benchmark format)."""
        docs: List[Document] = []
        chunks: List[Chunk] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                doc_id = item.get("doc_id", "DOC")
                text = item.get("text", "")
                title = item.get("title", doc_id)
                roles = item.get("access_roles", ["public"])
                dept = item.get("department", "General")

                doc = Document(
                    doc_id=doc_id,
                    title=title,
                    content=text,
                    access_roles=roles,
                    department=dept,
                )
                docs.append(doc)

                # For short benchmark documents, 1 doc = 1 chunk
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chk_1",
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    section="Main",
                    page=1,
                    access_roles=roles,
                    department=dept,
                )
                chunks.append(chunk)

        return docs, chunks
