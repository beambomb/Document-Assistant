"""LLM Generation engine supporting OpenRouter, Gemini, and Deterministic Mock."""

import re
import time
from pathlib import Path
from typing import List, Optional
import httpx
from src.core.models import RetrievalCandidate, QueryResult, Citation
from config.settings import settings

ENTERPRISE_SYSTEM_PROMPT = """Anda adalah Asisten Dokumen Resmi PT Majelis Debu Bintang.
Tugas Anda adalah menjawab pertanyaan pengguna secara lugas, padat, dan akurat HANYA berdasarkan dokumen yang diberikan.

ATURAN WAJIB:
1. Jawab langsung inti pertanyaan tanpa mengulang pertanyaan pengguna dan tanpa menyalin format template dokumen.
2. Setiap kalimat yang memuat fakta wajib menyertakan sitasi dokumen di akhir kalimat dalam format: [Doc: <ID_DOKUMEN>, Hal: <NOMOR_HALAMAN>] atau [Doc: <ID_DOKUMEN>].
3. Dilarang mengulang-ulang kalimat atau frasa yang sama. Susun jawaban dalam bahasa Indonesia yang alami dan profesional.
4. Jika informasi tidak ada dalam dokumen resmi yang tersedia, nyatakan: "Maaf, informasi ini tidak ditemukan dalam dokumen resmi yang tersedia untuk akun Anda."
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    def format_context(self, chunks: List[RetrievalCandidate]) -> str:
        context_blocks = []
        for i, c in enumerate(chunks, start=1):
            chunk = c.chunk
            block = f"[{chunk.doc_id} - {chunk.title}, Hal. {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/beambomb/Document-Assistant",
            "X-Title": "Enterprise Document Assistant",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        raw = msg.get("content") or msg.get("reasoning") or ""
                        # Clean out reasoning tags like <think>...</think> if present
                        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                        return cleaned if cleaned else raw.strip()
                    return "Tidak ada respon dari penyedia AI."
                else:
                    return (
                        f"Penyedia AI ({self.model}) mengembalikan error ({resp.status_code}): {resp.text[:120]}. "
                        f"Gunakan fallback dokumen: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
                    )
        except Exception as e:
            return MockGenerator().generate(query, retrieved_chunks, user_roles)


_SHARED_GGUF_INSTANCES = {}


class LocalGGUFGenerator(BaseGenerator):
    """
    Offline local LLM generator using GGUF quantized models (llama-cpp-python).
    Operates 100% on CPU without requiring cloud API keys or external server processes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
    ):
        self.model_path = model_path or settings.local_model_path
        self.n_ctx = n_ctx or settings.local_model_ctx
        self.n_threads = n_threads or settings.local_model_threads
        self.temperature = temperature if temperature is not None else 0.3
        self.max_tokens = max_tokens if max_tokens is not None else min(settings.max_tokens, 350)
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else 1.22

    def _load_model(self):
        global _SHARED_GGUF_INSTANCES
        cache_key = (str(self.model_path), self.n_ctx, self.n_threads)
        if cache_key not in _SHARED_GGUF_INSTANCES:
            try:
                from llama_cpp import Llama
                _SHARED_GGUF_INSTANCES[cache_key] = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_batch=512,
                    verbose=False,
                )
            except ImportError:
                raise RuntimeError(
                    "Engine GGUF lokal membutuhkan paket 'llama-cpp-python'.\n"
                    "Silakan install dengan perintah: pip install llama-cpp-python"
                )
        return _SHARED_GGUF_INSTANCES[cache_key]

    def format_context(self, chunks: List[RetrievalCandidate]) -> str:
        context_blocks = []
        for i, c in enumerate(chunks, start=1):
            chunk = c.chunk
            block = f"[{chunk.doc_id} - {chunk.title}, Hal. {chunk.page}]\n{chunk.text}"
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

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
                f"Silakan unduh model GGUF dan letakkan di folder 'models/'. "
                f"Gunakan fallback dokumen: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
            )

        try:
            llm = self._load_model()
            context_str = self.format_context(retrieved_chunks)
            stop_tokens = [
                "<|im_end|>", "<|im_start|>", "<|endoftext|>", "###",
                "\nPERTANYAAN:", "\nDOKUMEN:"
            ]

            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": ENTERPRISE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"DOKUMEN RESMI:\n{context_str}\n\nPERTANYAAN: {query}\n\nJAWABAN LANGSUNG:"},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                repeat_penalty=self.repeat_penalty,
                frequency_penalty=0.35,
                presence_penalty=0.2,
                stop=stop_tokens,
            )
            raw_answer = resp["choices"][0]["message"]["content"].strip()
            for prefix in ["JAWABAN LANGSUNG:", "JAWABAN RESMI:", "JAWABAN:", "Jawaban:"]:
                if raw_answer.startswith(prefix):
                    raw_answer = raw_answer[len(prefix):].strip()
            return raw_answer
        except Exception as e:
            return (
                f"Error eksekusi GGUF: {e}. "
                f"Gunakan fallback: {MockGenerator().generate(query, retrieved_chunks, user_roles)}"
            )


def get_generator(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    n_threads: Optional[int] = None,
    n_ctx: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    *args,
    **kwargs,
) -> BaseGenerator:
    """Factory creating configured generator instance."""
    active = (provider or kwargs.get("provider") or settings.llm_provider).lower().strip()
    active_model = model or kwargs.get("model")
    active_key = api_key or kwargs.get("api_key")
    active_temp = temperature if temperature is not None else kwargs.get("temperature")
    active_tokens = max_tokens if max_tokens is not None else kwargs.get("max_tokens")
    active_threads = n_threads if n_threads is not None else kwargs.get("n_threads")
    active_ctx = n_ctx if n_ctx is not None else kwargs.get("n_ctx")
    active_repeat = repeat_penalty if repeat_penalty is not None else kwargs.get("repeat_penalty")

    if active in ["gguf", "local"]:
        return LocalGGUFGenerator(
            model_path=active_model,
            n_ctx=active_ctx,
            n_threads=active_threads,
            temperature=active_temp,
            max_tokens=active_tokens,
            repeat_penalty=active_repeat,
        )
    if active == "openrouter":
        return OpenRouterGenerator(
            api_key=active_key,
            model=active_model,
            temperature=active_temp,
            max_tokens=active_tokens,
        )
    return MockGenerator()
