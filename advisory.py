from __future__ import annotations

import os
from typing import Optional


def try_get_llama():
    try:
        import importlib

        module = importlib.import_module("llama_cpp")
        return module.Llama
    except Exception:  # pragma: no cover
        return None


class LocalAdvisoryEngine:
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 1024):
        self.model_path = model_path or os.getenv("GGUF_MODEL_PATH")
        self.n_ctx = n_ctx
        self.llm = None
        llama_cls = try_get_llama()

        if self.model_path and llama_cls is not None:
            self.llm = llama_cls(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

    def _fallback(self, crop: str, disease: str, confidence: float, language: str) -> str:
        conf_pct = round(confidence * 100, 2)
        return (
            f"Prediction: {disease} ({conf_pct}%). "
            f"For {crop}, isolate affected leaves, avoid overhead irrigation, "
            "spray a crop-appropriate fungicide/pesticide as per local agricultural officer advice, "
            "and monitor spread for 3-5 days. "
            f"Requested language: {language}."
        )

    def generate(self, crop: str, disease: str, confidence: float, language: str = "English") -> str:
        if self.llm is None:
            return self._fallback(crop, disease, confidence, language)

        prompt = (
            "You are an agricultural assistant for farmers. "
            "Give concise, practical, safe advice in plain language.\n"
            f"Crop: {crop}\n"
            f"Disease prediction: {disease}\n"
            f"Confidence: {round(confidence * 100, 2)}%\n"
            f"Response language: {language}\n"
            "Include: immediate action, treatment options, and prevention tips."
        )

        output = self.llm(
            prompt,
            max_tokens=220,
            temperature=0.3,
            top_p=0.9,
            stop=["\n\nUser:", "\n\n###"],
        )
        text = output["choices"][0]["text"].strip()
        return text if text else self._fallback(crop, disease, confidence, language)
