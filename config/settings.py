"""Application settings and configuration management."""

from pathlib import Path
from typing import List, Dict
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App Information
    app_name: str = "Enterprise Document Assistant"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Security & Roles
    roles_config_path: Path = BASE_DIR / "config" / "roles.yaml"
    default_role: str = "employee"

    # Storage Paths
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    eval_goldens_dir: Path = BASE_DIR / "data" / "eval_goldens"
    indices_dir: Path = BASE_DIR / "data" / "indices"

    # Embedding & Dense Search
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    use_mock_embeddings_for_fast_test: bool = False

    # Retrieval & Reranking Parameters
    top_k_bm25: int = 10
    top_k_dense: int = 10
    top_k_fused: int = 10
    top_k_final: int = 3
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5

    # Reranker & Confidence Gate
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True
    confidence_threshold: float = 0.35  # Scores below this will trigger abstention

    # LLM Settings (Supports 'openrouter', 'gemini', 'openai', and 'mock')
    llm_provider: str = Field(
        default="openrouter",
        description="Active LLM provider: 'openrouter', 'gemini', 'openai', or 'mock' (for automated tests)"
    )

    # 1. OpenRouter API (Primary Cloud Provider)
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # 2. Google Gemini API (Alternative)
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-1.5-flash"

    # 3. Direct OpenAI API (Alternative)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # 4. Local Offline GGUF Model (llama-cpp-python / ctransformers)
    local_model_path: str = Field(default="models/llama-3.2-1b-instruct.Q4_K_M.gguf", validation_alias="LOCAL_MODEL_PATH")
    local_model_threads: int = 4
    local_model_ctx: int = 2048

    # Generation Parameters
    temperature: float = 0.0
    max_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def load_role_hierarchy(self) -> Dict[str, List[str]]:
        """Loads allowed access roles for each user role from roles.yaml."""
        if not self.roles_config_path.exists():
            return {
                "guest": ["public"],
                "employee": ["public", "internal", "employee"],
                "finance": ["public", "internal", "employee", "finance"],
                "hr": ["public", "internal", "employee", "hr"],
                "management": ["public", "internal", "employee", "finance", "hr", "management", "confidential"],
                "admin": ["public", "internal", "employee", "finance", "hr", "management", "confidential", "admin"],
            }
        with open(self.roles_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            roles_dict = {}
            for role_name, role_info in data.get("roles", {}).items():
                roles_dict[role_name] = role_info.get("allowed_access_roles", [role_name])
            return roles_dict


settings = Settings()
