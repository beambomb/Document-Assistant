"""LLM Generation engine supporting OpenRouter, Gemini, and Deterministic Mock."""

import re
import time
from typing import List, Optional
import httpx
from src.core.models import RetrievalCandidate, QueryResult, Citation
from config.settings import settings

ENTERPRISE_SYSTEM_PROMPT = """Anda adalah Asisten Dokumen Resmi Perusahaan PT Dua Empat Tujuh Nusantara Technology.
Tugas Anda adalah menjawab pertanyaan pengguna secara lugas, akurat, dan profesional HANYA berdasarkan dokumen konteks yang diberikan di bawah ini.

ATURAN WAJIB (STRICT GUARDRAILS):
1. HANYA gunakan fakta yang tertulis eksplisit dalam konteks dokumen. Dilarang keras berasumsi atau menggunakan pengetahuan umum eksternal.
2. Setiap kalimat yang memuat fakta atau angka WAJIB menyertakan tag sitasi di akhir kalimat dalam format: [Doc: <ID_DOKUMEN>, Hal: <NOMOR_HALAMAN>] atau [Doc: <ID_DOKUMEN>].
3. Jika informasi yang ditanyakan sama sekali tidak ada di dalam konteks dokumen, Anda WAJIB menyatakan:
   "Maaf, informasi ini tidak ditemukan dalam dokumen resmi yang tersedia untuk akun Anda."
"""


class BaseGenerator:
    """Abstract base class for LLM generation."""

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievalCandidate],
        user_roles: List[str],
    ) -> str:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Deterministic mock generator for zero-cost offline tests and CI/CD runners."""

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievalCandidate],
        user_roles: List[str],
    ) -> str:
        if not retrieved_chunks:
            return "Maaf, informasi ini tidak ditemukan dalam dokumen resmi yang tersedia untuk akun Anda."

        top = retrieved_chunks[0].chunk
        # Synthesize a deterministic grounded sentence citing top chunk
        return (
            f"Berdasarkan {top.title} ({top.section}), {top.text[:180]}... "
            f"[Doc: {top.doc_id}, Hal: {top.page}]"
        )


class OpenRouterGenerator(BaseGenerator):
    """Cloud LLM generator using OpenRouter API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")

    def format_context(self, chunks: List[RetrievalCandidate]) -> str:
        context_blocks = []
        for i, c in enumerate(chunks, start=1):
            chunk = c.chunk
            block = (
                f"--- DOKUMEN #{i} ---\n"
                f"ID Dokumen: {chunk.doc_id}\n"
                f"Judul: {chunk.title}\n"
                f"Bagian: {chunk.section} (Halaman {chunk.page})\n"
                f"Isi Dokumen:\n{chunk.text}\n"
            )
            context_blocks.append(block)
        return "\n".join(context_blocks)

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievalCandidate],
        user_roles: List[str],
    ) -> str:
        if not self.api_key:
            # Fallback to mock if API key is missing
            return MockGenerator().generate(query, retrieved_chunks, user_roles)

        context_str = self.format_context(retrieved_chunks)
        user_content = f"KONTEKS DOKUMEN:\n{context_str}\n\nPERTANYAAN PENGGUNA:\n{query}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ENTERPRISE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/beambomb/Document-Assistant",
            "X-Title": "Enterprise Document Assistant",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    return (
                        f"Penyedia AI mengembalikan error ({resp.status_code}). "
                        f"Gunakan fallback dokumen: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
                    )
        except Exception as e:
            return MockGenerator().generate(query, retrieved_chunks, user_roles)


def get_generator() -> BaseGenerator:
    """Factory creating configured generator instance."""
    provider = settings.llm_provider.lower().strip()
    if provider == "openrouter":
        return OpenRouterGenerator()
    return MockGenerator()
