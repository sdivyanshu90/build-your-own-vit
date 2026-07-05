"""Serving configuration sourced from environment variables.

Follows the Twelve-Factor "config in the environment" principle. Every field
maps to a ``VIT_``-prefixed environment variable (see ``.env.example``), so the
same image runs in any environment purely by changing env vars — no code or
file changes.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """Runtime configuration for the inference API."""

    model_config = SettingsConfigDict(
        env_prefix="VIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- model / inference ---
    model_checkpoint: str | None = Field(
        default=None, description="Path to a trained checkpoint (.pt)."
    )
    device: str = Field(default="auto")
    num_threads: int = Field(default=0, ge=0)
    use_ema: bool = Field(default=True)

    # --- HTTP server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    root_path: str = Field(default="")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    # --- security / limits ---
    cors_origins: str = Field(default="", description="Comma-separated CORS origins.")
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    rate_limit_rpm: int = Field(default=120, ge=0)
    api_token: str | None = Field(default=None, description="Optional bearer token.")

    # --- observability ---
    enable_metrics: bool = Field(default=True)
    default_top_k: int = Field(default=5, ge=1)

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse ``cors_origins`` into a clean list of origins."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
