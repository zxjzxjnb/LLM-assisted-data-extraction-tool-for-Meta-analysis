"""Configuration for the model used by the paper locator.

Profiles deliberately describe transport details rather than model vendors.  A
profile can therefore point at Gemini, an OpenAI-compatible hosted endpoint, or
an OpenAI-compatible service running on the local network.  Secrets are read
from the environment by default; a literal ``api_key`` is accepted for tests
and for callers that already manage secrets outside this package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ResponseMode = Literal["auto", "json_schema", "json_object"]
Transport = Literal["gemini", "openai_compatible"]


class ModelProfile(BaseModel):
    """One locator endpoint and its optional model fallback references."""

    transport: Transport = "gemini"
    # ``base_url`` is accepted alongside the more user-facing ``endpoint``
    # spelling used in the design document.
    endpoint: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    response_mode: ResponseMode = "auto"
    fallbacks: list[str | dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_endpoint_alias(self) -> "ModelProfile":
        if self.endpoint and self.base_url and self.endpoint.rstrip("/") != self.base_url.rstrip("/"):
            raise ValueError("endpoint and base_url must refer to the same URL when both are set")
        if self.endpoint:
            self.endpoint = self.endpoint.rstrip("/")
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")
        return self

    @property
    def resolved_endpoint(self) -> str | None:
        return self.endpoint or self.base_url

    @property
    def resolved_api_key_env(self) -> str:
        if self.api_key_env:
            return self.api_key_env
        return "GOOGLE_API_KEY" if self.transport == "gemini" else "OPENAI_API_KEY"

    def api_key_value(self) -> str | None:
        return self.api_key or os.environ.get(self.resolved_api_key_env)

    def fallback_names(self) -> list[str]:
        """Normalize ``["backup"]`` and ``[{"profile": "backup"}]`` forms."""
        names: list[str] = []
        for item in self.fallbacks:
            name = item if isinstance(item, str) else item.get("profile")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names


class ModelProfileConfig(BaseModel):
    """A named profile collection loaded from a small YAML file."""

    active_profile: str = "default"
    profiles: dict[str, ModelProfile] = Field(default_factory=dict)

    @classmethod
    def default(cls) -> "ModelProfileConfig":
        """Return the backwards-compatible Gemini configuration.

        No fallback is included here.  This is intentional: a model switch can
        change location results, so it must be requested explicitly in config.
        """
        return cls(
            active_profile="default",
            profiles={
                "default": ModelProfile(
                    transport="gemini",
                    model=os.environ.get("METAEXTRACT_MODEL", "gemini-2.5-pro"),
                    api_key_env="GOOGLE_API_KEY",
                    response_mode="auto",
                )
            },
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelProfileConfig":
        import yaml

        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"model profile config must be a mapping: {path}")
        # A profile-only mapping is convenient for small local config files.
        if "profiles" not in data:
            profile_data = data
            first_profile = next(iter(profile_data), "default")
            data = {"active_profile": first_profile, "profiles": profile_data}
        config = cls.model_validate(data)
        if not config.profiles:
            raise ValueError(f"model profile config has no profiles: {path}")
        if config.active_profile not in config.profiles:
            raise ValueError(
                f"active_profile {config.active_profile!r} is not defined in {path}"
            )
        return config

    def profile(self, name: str | None = None) -> ModelProfile:
        selected = name or self.active_profile
        try:
            return self.profiles[selected]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles)) or "(none)"
            raise ValueError(f"unknown model profile {selected!r}; available: {available}") from exc

    def chain(self, name: str | None = None, extra_fallbacks: list[str] | None = None) -> list[tuple[str, ModelProfile]]:
        """Resolve the active profile and its explicit fallback chain.

        Cycles and duplicate names are rejected before any network call.  An
        empty fallback list means exactly one attempted profile.
        """
        first = name or self.active_profile
        pending = [first]
        if extra_fallbacks:
            pending.extend(extra_fallbacks)
        seen: set[str] = set()
        result: list[tuple[str, ModelProfile]] = []
        while pending:
            current = pending.pop(0)
            if current in seen:
                raise ValueError(f"model profile fallback cycle or duplicate: {current!r}")
            seen.add(current)
            profile = self.profile(current)
            result.append((current, profile))
            pending.extend(profile.fallback_names())
        return result


def load_model_profiles(path: str | Path | None = None) -> ModelProfileConfig:
    """Load the configured profiles, or return the safe Gemini default."""
    if path is None:
        configured = os.environ.get("METAEXTRACT_MODEL_PROFILES")
        path = configured or Path(__file__).resolve().parents[2] / "data" / "model_profiles.yaml"
    path = Path(path)
    if path.exists():
        return ModelProfileConfig.from_yaml(path)
    # An explicitly requested path should fail loudly; the conventional default
    # remains optional for existing checkouts.
    if os.environ.get("METAEXTRACT_MODEL_PROFILES") or str(path) != str(
        Path(__file__).resolve().parents[2] / "data" / "model_profiles.yaml"
    ):
        raise FileNotFoundError(f"model profile config does not exist: {path}")
    return ModelProfileConfig.default()

