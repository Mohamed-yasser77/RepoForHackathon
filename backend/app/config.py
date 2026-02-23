"""
Application Configuration — Milestone 3
========================================
Centralised settings via pydantic-settings.
All values are overridable via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Immutable application configuration loaded from environment."""

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL for ARQ broker and Redlock",
    )

    # ── Redlock ───────────────────────────────────────────────────────────
    REDLOCK_TTL_MS: int = Field(
        default=5000,
        description="Distributed lock TTL in milliseconds",
    )
    REDLOCK_KEY_PREFIX: str = Field(
        default="ticket_lock:",
        description="Key prefix for Redlock resources",
    )

    # ── Webhook ───────────────────────────────────────────────────────────
    WEBHOOK_URL: str = Field(
        default="https://httpbin.org/post",
        description="Mock Discord/Slack webhook URL for urgent ticket alerts",
    )

    # ── Urgency Threshold ─────────────────────────────────────────────────
    URGENCY_THRESHOLD: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Regression score above which a webhook alert is triggered",
    )

    # ── ARQ ────────────────────────────────────────────────────────────────
    ARQ_QUEUE_NAME: str = Field(
        default="arq:queue",
        description="ARQ task queue name",
    )

    # ── Milestone 3: Flash-Flood / Semantic Dedup ───────────────────────────
    FLASH_FLOOD_WINDOW_SECS: int = Field(
        default=300,
        description="Redis rolling window (seconds) for similarity lookups",
    )
    FLASH_FLOOD_SIM_THRESHOLD: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for considering tickets similar",
    )
    FLASH_FLOOD_COUNT_THRESHOLD: int = Field(
        default=10,
        description="Number of similar tickets within window that triggers a flood",
    )

    # ── Milestone 3: Circuit Breaker ─────────────────────────────────────────
    CIRCUIT_BREAKER_LATENCY_MS: int = Field(
        default=500,
        description="EWMA latency ceiling (ms) before circuit trips to OPEN",
    )
    CIRCUIT_BREAKER_COOLDOWN_SECS: int = Field(
        default=30,
        description="Seconds in OPEN state before attempting HALF_OPEN probe",
    )
    M3_FALLBACK_ONLY: bool = Field(
        default=False,
        description="If True, skip loading Transformer models and use M1 fallback immediately (RAM saving)",
    )

    # ── Server ────────────────────────────────────────────────────────────
    APP_NAME: str = "Smart-Support Milestone 3"
    APP_VERSION: str = "3.0.0"
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: str = Field(
        default="",
        description="Comma-separated list of allowed CORS origins",
    )
    CORS_ALLOWED_ORIGINS: str = Field(
        default="",
        description="Comma-separated list of allowed CORS origins",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = Settings()
