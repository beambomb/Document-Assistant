"""LLM Generation engine supporting OpenRouter, Gemini, and Deterministic Mock."""

import re
import time
from pathlib import Path
from typing import List, Optional
import httpx
from src.core.models import RetrievalCandidate, QueryResult, Citation
from config.settings import settings

ENTERPRISE_SYSTEM_PROMPT = """Anda adalah Asisten Dokumen Resmi PT Majelis Debu Bintang.
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


_SHARED_GGUF_INSTANCE = None


class LocalGGUFGenerator(BaseGenerator):
    """
    Offline local LLM generator using GGUF quantized models (llama-cpp-python or ctransformers).
    Operates 100% on CPU without requiring cloud API keys or external server processes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
    ):
        self.model_path = model_path or settings.local_model_path
        self.n_ctx = n_ctx or settings.local_model_ctx
        self.n_threads = n_threads or settings.local_model_threads

    def _load_model(self):
        global _SHARED_GGUF_INSTANCE
        if _SHARED_GGUF_INSTANCE is None:
            # Try llama_cpp first, then ctransformers
            try:
                from llama_cpp import Llama
                _SHARED_GGUF_INSTANCE = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
            except ImportError:
                try:
                    from ctransformers import AutoModelForCausalLM
                    _SHARED_GGUF_INSTANCE = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        model_type="llama",
                        threads=self.n_threads,
                    )
                except ImportError:
                    raise RuntimeError(
                        "Untuk menjalankan model GGUF lokal di Python, silakan install salah satu engine:\n"
                        "  pip install llama-cpp-python\n"
                        "atau\n"
                        "  pip install ctransformers"
                    )
        return _SHARED_GGUF_INSTANCE

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
        if not retrieved_chunks:
            return "Maaf, informasi ini tidak ditemukan dalam dokumen resmi yang tersedia untuk akun Anda."

        path_obj = Path(self.model_path)
        if not path_obj.exists():
            return (
                f"[GGUF Engine Status]: File model lokal belum ditemukan di '{self.model_path}'. "
                f"Silakan unduh model GGUF (contoh: Llama-3.2-1B-Instruct-Q4_K_M.gguf) dan letakkan di folder 'models/'. "
                f"Gunakan fallback dokumen: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
            )

        try:
            llm = self._load_model()
            context_str = self.format_context(retrieved_chunks)
            gen_tokens = min(settings.max_tokens, 256)
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "###"]

            if hasattr(llm, "create_chat_completion"):
                # llama-cpp chat interface
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": ENTERPRISE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"KONTEKS DOKUMEN:\n{context_str}\n\nPERTANYAAN PENGGUNA:\n{query}"},
                    ],
                    max_tokens=gen_tokens,
                    temperature=settings.temperature,
                    stop=stop_tokens,
                )
                return resp["choices"][0]["message"]["content"].strip()
            else:
                # ctransformers text interface
                prompt = (
                    f"### SYSTEM:\n{ENTERPRISE_SYSTEM_PROMPT}\n\n"
                    f"### USER:\nKONTEKS DOKUMEN:\n{context_str}\n\nPERTANYAAN PENGGUNA:\n{query}\n\n"
                    f"### ASSISTANT:\n"
                )
                return llm(prompt, max_new_tokens=gen_tokens).strip()
        except Exception as e:
            return (
                f"Error eksekusi GGUF: {e}. "
                f"Gunakan fallback: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
            )


def get_generator() -> BaseGenerator:
    """Factory creating configured generator instance."""
    provider = settings.llm_provider.lower().strip()
    if provider in ["gguf", "local"]:
        return LocalGGUFGenerator()
    if provider == "openrouter":
        return OpenRouterGenerator()
    return MockGenerator()
