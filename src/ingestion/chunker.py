"""Markdown and text chunker with hierarchical metadata and access tag inheritance."""

import re
from typing import List, Dict, Any, Tuple
from src.core.models import Chunk, Document


class DocumentChunker:
    """Chunks documents while preserving section hierarchy and ACL tags."""

    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_markdown_metadata(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Extracts top metadata block from markdown document."""
        metadata: Dict[str, Any] = {
            "doc_id": "UNKNOWN",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "access_roles": ["internal"],
            "department": "General",
        }

        # Regex patterns for standard enterprise headers
        patterns = {
            "doc_id": r"\*\*Nomor Dokumen\*\*:\s*([^\n\r]+)",
            "version": r"\*\*Versi\*\*:\s*([^\n\r]+)",
            "effective_date": r"\*\*Tanggal Berlaku\*\*:\s*([^\n\r]+)",
            "access_roles": r"\*\*Klasifikasi Akses\*\*:\s*([^\n\r]+)",
            "department": r"\*\*Divisi\*\*:\s*([^\n\r]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if key == "access_roles":
                    # Parse comma-separated roles
                    roles = [r.strip().lower() for r in re.split(r"[,;]", val) if r.strip()]
                    metadata["access_roles"] = roles
                else:
                    metadata[key] = val

        # Extract title from first # Header if available
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            metadata["title"] = metadata["doc_id"]

        return metadata, text

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunks a Document object into structured, metadata-enriched Chunks."""
        chunks: List[Chunk] = []
        raw_text = doc.content

        # Split document by markdown level-2 headers: ## Section Name
        section_pattern = re.compile(r"(^##\s+.+$)", re.MULTILINE)
        splits = section_pattern.split(raw_text)

        current_section = "Pendahuluan"
        section_texts: List[Tuple[str, str]] = []

        if splits:
            i = 0
            while i < len(splits):
                part = splits[i].strip()
                if part.startswith("## "):
                    current_section = part.replace("## ", "").strip()
                    i += 1
                    if i < len(splits):
                        section_body = splits[i].strip()
                        section_texts.append((current_section, section_body))
                        i += 1
                else:
                    if part:
                        section_texts.append((current_section, part))
                    i += 1
        else:
            section_texts = [(current_section, raw_text)]

        # Generate chunks per section
        chunk_counter = 1
        page_counter = 1

        for sec_name, sec_body in section_texts:
            if not sec_body:
                continue

            # Remove markdown header artifacts from body
            clean_body = re.sub(r"^\*\*[^\*]+\*\*:[^\n\r]+\n?", "", sec_body, flags=re.MULTILINE).strip()
            if not clean_body:
                continue

            # Check if section fits in one chunk or needs paragraph sub-chunking
            paragraphs = [p.strip() for p in clean_body.split("\n\n") if p.strip()]

            current_chunk_text = ""
            for p in paragraphs:
                if len(current_chunk_text) + len(p) > self.target_chunk_size and current_chunk_text:
                    chunk_id = f"{doc.doc_id}_chk_{chunk_counter}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            title=doc.title,
                            text=current_chunk_text.strip(),
                            section=sec_name,
                            page=page_counter,
                            access_roles=doc.access_roles,
                            version=doc.version,
                            effective_date=doc.effective_date,
                            department=doc.department,
                        )
                    )
                    chunk_counter += 1
                    if chunk_counter % 2 == 0:
                        page_counter += 1
                    current_chunk_text = p + "\n\n"
                else:
                    current_chunk_text += p + "\n\n"

            if current_chunk_text.strip():
                chunk_id = f"{doc.doc_id}_chk_{chunk_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=current_chunk_text.strip(),
                        section=sec_name,
                        page=page_counter,
                        access_roles=doc.access_roles,
                        version=doc.version,
                        effective_date=doc.effective_date,
                        department=doc.department,
                    )
                )
                chunk_counter += 1

        return chunks
