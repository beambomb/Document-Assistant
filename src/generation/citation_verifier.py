"""Citation verifier checking grounding against retrieved chunks."""

import re
from typing import List, Tuple, Set
from src.core.models import Citation, RetrievalCandidate


class CitationVerifier:
    """Extracts and verifies citations in generated responses."""

    # Matches [Doc: DOC-ID, Hal: 1] or [Doc: DOC-ID]
    CITATION_PATTERN = re.compile(
        r"\[Doc:\s*([A-Za-z0-9_-]+)(?:,\s*Hal:\s*(\d+))?\]",
        re.IGNORECASE,
    )

    def extract_and_verify(
        self,
        answer: str,
        retrieved_chunks: List[RetrievalCandidate],
    ) -> Tuple[List[Citation], float, int]:
        """
        Extracts citations from generated text and checks their presence in retrieved chunks.

        Returns:
            citations: List of verified Citation objects
            precision: ratio of valid citations to total citations (1.0 = perfect)
            hallucination_count: count of fabricated citations
        """
        matches = self.CITATION_PATTERN.findall(answer)
        if not matches:
            return [], 0.0, 0

        # Build lookup table of retrieved doc_ids and chunks
        retrieved_doc_map = {}
        for c in retrieved_chunks:
            retrieved_doc_map[c.chunk.doc_id.lower()] = c.chunk

        citations: List[Citation] = []
        valid_count = 0
        hallucination_count = 0

        for doc_id_match, page_match in matches:
            norm_doc_id = doc_id_match.strip().lower()
            page_num = int(page_match) if page_match else 1

            if norm_doc_id in retrieved_doc_map:
                valid_count += 1
                matched_chunk = retrieved_doc_map[norm_doc_id]
                citations.append(
                    Citation(
                        doc_id=matched_chunk.doc_id,
                        chunk_id=matched_chunk.chunk_id,
                        title=matched_chunk.title,
                        section=matched_chunk.section,
                        page=page_num if page_match else matched_chunk.page,
                        quote=matched_chunk.text[:120],
                    )
                )
            else:
                hallucination_count += 1
                # Hallucinated citation: cited a doc that was NOT in the context
                citations.append(
                    Citation(
                        doc_id=doc_id_match,
                        chunk_id="HALLUCINATED",
                        title="Dokumen Tidak Terverifikasi",
                        section="Unknown",
                        page=page_num,
                        quote="",
                    )
                )

        precision = valid_count / len(matches) if matches else 0.0
        return citations, precision, hallucination_count
