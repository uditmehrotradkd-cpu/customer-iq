"""Environment-driven runtime settings (prefix ``SEG_``)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from segmentation.config import ARTIFACTS_DIR


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _first_env(names) -> str:
    return next((os.environ[n] for n in names if os.getenv(n)), "")


LLM_PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.8-flash",
        "fallbacks": "gemini-3.7-flash,gemini-3.5-flash,gemini-flash-latest",
        "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "reasoning": "low",
    },
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": ("OPENAI_API_KEY",)},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key_env": ("GROQ_API_KEY",)},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen3.5:4b", "key_env": (), "reasoning": "none"},
}


def _llm_provider() -> str:
    explicit = os.getenv("SEG_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if "SEG_LLM_BASE_URL" in os.environ:
        return "custom"
    key = os.getenv("SEG_LLM_API_KEY", "")
    if key.startswith(("sk-", "gsk_")):
        return "groq" if key.startswith("gsk_") else "openai"
    for name in ("gemini", "openai", "groq"):
        if _first_env(LLM_PROVIDERS[name]["key_env"]):
            return name
    return "gemini"


@dataclass(frozen=True)
class Settings:
    app_name: str = "Customer Segmentation"
    version: str = "1.0.0"
    artifacts_dir: Path = field(default_factory=lambda: Path(os.getenv("SEG_ARTIFACTS_DIR", str(ARTIFACTS_DIR))))
    auto_train: bool = field(default_factory=lambda: _env_bool("SEG_AUTO_TRAIN", True))
    allowed_hosts: list[str] = field(default_factory=lambda: _env_list("SEG_ALLOWED_HOSTS", "*"))
    cors_origins: list[str] = field(default_factory=lambda: _env_list("SEG_CORS_ORIGINS", ""))
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("SEG_MAX_UPLOAD_MB", "5")))
    max_batch_rows: int = field(default_factory=lambda: int(os.getenv("SEG_MAX_BATCH_ROWS", "50000")))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("SEG_RATE_LIMIT_PER_MINUTE", "60")))
    enable_hsts: bool = field(default_factory=lambda: _env_bool("SEG_ENABLE_HSTS", False))
    enable_docs: bool = field(default_factory=lambda: _env_bool("SEG_ENABLE_DOCS", True))
    log_level: str = field(default_factory=lambda: os.getenv("SEG_LOG_LEVEL", "INFO").upper())
    # Where agent drafts and published models are stored (must be writable; mount a volume to persist).
    model_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SEG_MODEL_STORE_DIR", str(Path(os.getenv("SEG_ARTIFACTS_DIR", str(ARTIFACTS_DIR))) / "store")))
    )
    max_train_rows: int = field(default_factory=lambda: int(os.getenv("SEG_MAX_TRAIN_ROWS", "20000")))
    min_train_rows: int = field(default_factory=lambda: int(os.getenv("SEG_MIN_TRAIN_ROWS", "150")))
    train_limit_per_10min: int = field(default_factory=lambda: int(os.getenv("SEG_TRAIN_LIMIT_PER_10MIN", "6")))
    login_limit_per_10min: int = field(default_factory=lambda: int(os.getenv("SEG_LOGIN_LIMIT_PER_10MIN", "20")))
    # Cloud PostgreSQL for accounts and chat history (Railway, Neon, Supabase...); empty = local SQLite file.
    database_url: str = field(default_factory=lambda: os.getenv("SEG_DATABASE_URL") or os.getenv("DATABASE_URL", ""), repr=False)
    # General AI answers through any OpenAI-compatible chat API. SEG_LLM_PROVIDER picks sensible defaults:
    # gemini (default; key in SEG_LLM_API_KEY or GEMINI_API_KEY), openai, groq, ollama. SEG_LLM_BASE_URL/MODEL override.
    llm_provider: str = field(default_factory=lambda: _llm_provider())
    llm_api_key: str = field(default_factory=lambda: os.getenv("SEG_LLM_API_KEY") or _first_env(LLM_PROVIDERS.get(_llm_provider(), {}).get("key_env", ())), repr=False)
    llm_base_url: str = field(default_factory=lambda: os.getenv("SEG_LLM_BASE_URL") or LLM_PROVIDERS.get(_llm_provider(), LLM_PROVIDERS["openai"])["base_url"])
    llm_model: str = field(default_factory=lambda: os.getenv("SEG_LLM_MODEL") or LLM_PROVIDERS.get(_llm_provider(), LLM_PROVIDERS["openai"])["model"])
    # Tried in order when the main model is overloaded or rate-limited.
    llm_fallback_models: list[str] = field(default_factory=lambda: _env_list("SEG_LLM_FALLBACK_MODELS", LLM_PROVIDERS.get(_llm_provider(), {}).get("fallbacks", "")))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("SEG_LLM_TIMEOUT", "30")))
    llm_reasoning_effort: str = field(default_factory=lambda: os.getenv("SEG_LLM_REASONING", LLM_PROVIDERS.get(_llm_provider(), {}).get("reasoning", "")))
    llm_limit_per_10min: int = field(default_factory=lambda: int(os.getenv("SEG_LLM_LIMIT_PER_10MIN", "40")))
    # A key, or an explicitly configured (e.g. local, keyless) endpoint, turns the AI on.
    llm_endpoint_configured: bool = field(default_factory=lambda: "SEG_LLM_BASE_URL" in os.environ or _llm_provider() == "ollama")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key) or self.llm_endpoint_configured

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def get_settings() -> Settings:
    return Settings()
