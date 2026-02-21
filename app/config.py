"""
Application Configuration — Milestone 2
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

    # ── Server ────────────────────────────────────────────────────────────
    APP_NAME: str = "Smart-Support Milestone 2"
    APP_VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = Settings()
