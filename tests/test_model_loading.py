from types import SimpleNamespace

import pytest
import torch

import llm_decision_engine.engine as engine_module
import llm_decision_engine.models as models_module
from llm_decision_engine import DecisionEngine


class FakeLoadedModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def test_decision_engine_requires_model():
    with pytest.raises(TypeError):
        DecisionEngine()


def test_decision_engine_accepts_local_path_and_forwards_local_only(
    monkeypatch,
):
    calls = {}

    bundle = SimpleNamespace(
        tokenizer=object(),
        model=object(),
    )

    def fake_load_model(
        model_id,
        *,
        local_files_only=False,
    ):
        calls["model_id"] = model_id
        calls["local_files_only"] = local_files_only
        return bundle

    monkeypatch.setattr(
        engine_module,
        "load_model",
        fake_load_model,
    )

    local_path = r"D:\models\Qwen2.5-3B-Instruct"

    engine = DecisionEngine(
        model=local_path,
        local_files_only=True,
    )

    assert calls == {
        "model_id": local_path,
        "local_files_only": True,
    }
    assert engine.model_id == local_path
    assert engine.local_files_only is True
    assert engine.tokenizer is bundle.tokenizer
    assert engine.model is bundle.model


@pytest.mark.parametrize(
    "local_files_only",
    [False, True],
)
def test_load_model_forwards_local_files_only(
    monkeypatch,
    local_files_only,
):
    calls = {}
    tokenizer = object()
    fake_model = FakeLoadedModel()

    def fake_tokenizer_from_pretrained(
        model_id,
        **kwargs,
    ):
        calls["tokenizer"] = {
            "model_id": model_id,
            **kwargs,
        }
        return tokenizer

    def fake_model_from_pretrained(
        model_id,
        **kwargs,
    ):
        calls["model"] = {
            "model_id": model_id,
            **kwargs,
        }
        return fake_model

    monkeypatch.setattr(
        models_module.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    monkeypatch.setattr(
        models_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    bundle = models_module.load_model(
        "Qwen/Qwen2.5-3B-Instruct",
        local_files_only=local_files_only,
    )

    assert calls["tokenizer"] == {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "local_files_only": local_files_only,
    }

    assert calls["model"] == {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": local_files_only,
    }

    assert bundle.tokenizer is tokenizer
    assert bundle.model is fake_model
    assert fake_model.eval_called is True
