from types import SimpleNamespace

import pytest

from metaextract.locate import LocatorResponse, TaskSpec, TargetVar, locate
from metaextract.model_profiles import ModelProfile, ModelProfileConfig
from metaextract.records import ScreeningResult
from metaextract.sourcedoc import BBox, SourceBlock, SourceDoc


def _doc() -> SourceDoc:
    return SourceDoc(
        doc_id="paper1",
        n_pages=1,
        blocks=[
            SourceBlock(
                block_id="p1.b1",
                kind="paragraph",
                page=1,
                bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                text="N2O emission table",
            )
        ],
    )


def _spec() -> TaskSpec:
    return TaskSpec(domain="demo", target_variables=[TargetVar(name="N2O emission")])


class _Completions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_default_profile_has_no_fallback():
    config = ModelProfileConfig.default()
    assert config.chain() == [("default", config.profile())]
    assert config.profile().fallbacks == []


def test_yaml_profiles_accept_endpoint_and_fallback_mapping(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        """
active_profile: main
profiles:
  main:
    transport: openai_compatible
    endpoint: https://example.test/v1
    model: main-model
    api_key_env: MAIN_KEY
    response_mode: json_schema
    fallbacks:
      - profile: backup
  backup:
    transport: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    model: backup-model
    api_key_env: BACKUP_KEY
    response_mode: json_object
""",
        encoding="utf-8",
    )
    config = ModelProfileConfig.from_yaml(path)
    assert [name for name, _ in config.chain()] == ["main", "backup"]
    assert config.profile("main").resolved_endpoint == "https://example.test/v1"


def test_openai_compatible_locator_uses_json_schema_and_metadata():
    result = LocatorResponse(
        include=True,
        regions=[{"block_id": "p1.b1", "variable_name": "N2O emission"}],
    )
    completions = _Completions(result.model_dump_json())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = ModelProfile(
        transport="openai_compatible",
        endpoint="https://provider.test/v1",
        model="provider-model",
        api_key="test-key",
        response_mode="json_schema",
    )

    out = locate(_doc(), spec=_spec(), profile=profile, profile_name="main", client=client)

    request = completions.calls[0]
    assert request["model"] == "provider-model"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["name"] == "locator_response"
    assert out.profile == "main"
    assert out.provider == "openai_compatible"
    assert out.endpoint == "https://provider.test/v1"
    assert out.model == "provider-model"
    assert out.response_mode == "json_schema"
    assert out.schema_hash and out.prompt_hash


def test_auto_mode_retries_json_object_when_schema_is_rejected():
    result = LocatorResponse(include=True).model_dump_json()

    class RejectSchema(_Completions):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["response_format"]["type"] == "json_schema":
                raise RuntimeError("unsupported response_format")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=result))]
            )

    completions = RejectSchema(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = ModelProfile(
        transport="openai_compatible", model="provider-model", api_key="test-key"
    )
    out = locate(_doc(), spec=_spec(), profile=profile, client=client)
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
    ]
    assert out.response_mode == "json_object"


def test_profile_cycles_are_rejected():
    config = ModelProfileConfig(
        active_profile="a",
        profiles={
            "a": ModelProfile(fallbacks=["b"]),
            "b": ModelProfile(fallbacks=["a"]),
        },
    )
    with pytest.raises(ValueError, match="cycle or duplicate"):
        config.chain()


def test_prepare_profile_fallback_is_opt_in_and_records_reason():
    from scripts.prepare_paper import _locate_with_profiles
    from metaextract.locate import LocateOutput

    config = ModelProfileConfig(
        active_profile="main",
        profiles={
            "main": ModelProfile(transport="openai_compatible", model="main"),
            "backup": ModelProfile(transport="openai_compatible", model="backup"),
        },
    )
    calls = []

    def fake_locate(_doc, *, spec, profile, profile_name):
        calls.append(profile_name)
        if profile_name == "main":
            raise RuntimeError("rate limited")
        return LocateOutput(screening=ScreeningResult(paper_id="paper1", include=True), regions=[])

    with pytest.raises(RuntimeError, match="rate limited"):
        _locate_with_profiles(fake_locate, _doc(), _spec(), config)
    assert calls == ["main"]

    calls.clear()
    out = _locate_with_profiles(fake_locate, _doc(), _spec(), config, fallbacks=["backup"])
    assert calls == ["main", "backup"]
    assert out.fallback_used
    assert "rate limited" in out.fallback_reasons[0]
    assert "rate limited" in out.fallback_reason

