#!/usr/bin/env python3
"""
Layerwise next-token logit-difference trajectories for political-routing analysis.

This script computes pairwise refusal-vs-answer logit differences at each
transformer layer for matched prompt pairs from either:

- the paper's v1 corpus (24 CCP + 24 matched controls), or
- the adversarial-v3 corpus (32 CCP + 32 matched controls), or
- the expanded political-v2 corpus (120 CCP + 120 matched controls), or
- the safety-v1 / safety-v2 corpora (16 / 32 harmful + benign pairs)

Primary outputs:
- pair-level per-layer records
- layerwise summary with bootstrap confidence intervals
- topic-level summary

Optional:
- apply an existing activation-ablation vector during the forward pass
  (primarily for Qwen checkpoint reuse)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from political_prompts_adversarial import (
    ADVERSARIAL_CCP,
    ADVERSARIAL_PARALLEL,
    INTENSITY_LABELS,
    TOPIC_LABELS,
)
from political_prompts_v1 import POLITICAL_ANSWERED_PROMPTS, POLITICAL_REFUSED_PROMPTS
from political_prompts_v2 import (
    POLITICAL_ANSWERED_PROMPTS_V2,
    POLITICAL_REFUSED_PROMPTS_V2,
)
from probe_cross_validation import V1_CATEGORIES
from safety_prompts import SAFETY_BENIGN_PROMPTS, SAFETY_HARMFUL_PROMPTS
from safety_prompts_v2 import SAFETY_BENIGN_PROMPTS_V2, SAFETY_HARMFUL_PROMPTS_V2


SEED = 42
DEFAULT_REFUSAL_STRINGS: Tuple[str, ...] = (
    "I",
    "I'm",
    "Sorry",
    "cannot",
    "can't",
    "unable",
    "As",
    "Unfortunately",
)


@dataclass(frozen=True)
class PromptPair:
    pair_idx: int
    topic: str
    intensity: str
    ccp_prompt: str
    control_prompt: str


@dataclass(frozen=True)
class AblationSpec:
    layer_idx: int
    alpha: float
    vector: torch.Tensor
    source_path: str
    source_kind: str


@dataclass(frozen=True)
class DecisionPoint:
    meaningful_token_id: int
    prefix_token_ids: Tuple[int, ...]
    generated_token_ids: Tuple[int, ...]


class IdentityModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _git_sha7() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _sanitize_model_short(model_id: str) -> str:
    short = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", short).strip("_")


def _next_run_dir(base_dir: Path, model_id: str, corpus: str, seed: int) -> Path:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    stem = f"{date_str}_routinglogit_{_sanitize_model_short(model_id)}_{corpus}_s{seed}_{_git_sha7()}"
    for idx in range(1, 100):
        candidate = base_dir / f"{stem}_r{idx:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate run dir for stem={stem}")


def _resolve_device(model: Any = None) -> torch.device:
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _ensure_transformers_compat_shims() -> None:
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu

    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: True
    if not hasattr(_u, "is_torch_fx_available"):
        _u.is_torch_fx_available = _iu.is_torch_fx_available

    if not hasattr(_u, "is_flash_attn_greater_or_equal_2_10"):
        _u.is_flash_attn_greater_or_equal_2_10 = lambda: False
    if not hasattr(_iu, "is_flash_attn_greater_or_equal_2_10"):
        _iu.is_flash_attn_greater_or_equal_2_10 = _u.is_flash_attn_greater_or_equal_2_10


def _resolve_transformer_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    # Multimodal wrappers (Gemma-3): model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "encoder") and hasattr(inner.encoder, "layers"):
            return inner.encoder.layers
    raise AttributeError("Could not locate transformer layers on model")


def _layer_for_index(model: Any, layer_idx: int) -> Any:
    return _resolve_transformer_layers(model)[layer_idx]


def load_model_and_tokenizer(model_id: str, device_map: str = "auto") -> Tuple[Any, Any]:
    _ensure_transformers_compat_shims()
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoModel

    trust_prefixes = (
        "internlm/",
        "baichuan-inc/",
        "THUDM/",
        "zai-org/",
        "stepfun-ai/",
        "MiniMaxAI/",
        "moonshotai/",
        "openbmb/",
    )
    needs_trust = any(model_id.startswith(prefix) for prefix in trust_prefixes)

    print(f"  Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=needs_trust)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {"torch_dtype": torch.float16, "trust_remote_code": needs_trust}
    if model_id.startswith("openbmb/MiniCPM4.1-8B"):
        kwargs["attn_implementation"] = "eager"

    dev = _resolve_device()
    if dev.type == "cuda":
        kwargs["device_map"] = device_map
    else:
        kwargs["device_map"] = None
        kwargs["low_cpu_mem_usage"] = True

    print(f"  Loading model weights for {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError, ImportError):
            model = AutoModel.from_pretrained(model_id, **kwargs)

    if dev.type == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer


def _tokenize(
    tokenizer: Any,
    texts: Any,
    *,
    max_length: int = 256,
    padding: bool = False,
    chat_template: bool = True,
) -> Mapping[str, Any]:
    text_or_texts = texts
    if chat_template:
        source = [texts] if isinstance(texts, str) else list(texts)
        templated: List[str] = []
        for text in source:
            messages = [{"role": "user", "content": text}]
            kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            try:
                kwargs["enable_thinking"] = False
                templated.append(tokenizer.apply_chat_template(messages, **kwargs))
            except TypeError:
                kwargs.pop("enable_thinking", None)
                templated.append(tokenizer.apply_chat_template(messages, **kwargs))
            except ValueError:
                templated.append(text)
        text_or_texts = templated if len(templated) > 1 else templated[0]

    kwargs = {"return_tensors": "pt", "truncation": True, "max_length": max_length}
    if padding:
        kwargs["padding"] = True
    encoded = tokenizer(text_or_texts, **kwargs)
    if hasattr(encoded, "to"):
        encoded = dict(encoded)
    return encoded


def _default_generate_kwargs(tokenizer: Any) -> Dict[str, Any]:
    eos = getattr(tokenizer, "eos_token_id", None)
    kwargs = {
        "max_new_tokens": 96,
        "do_sample": False,
        "temperature": 0.0,
        "pad_token_id": eos,
    }
    if str(getattr(tokenizer, "name_or_path", "")).startswith("openbmb/MiniCPM4.1-8B"):
        kwargs["use_cache"] = False
    return kwargs


@contextmanager
def temporary_ablation(
    model: Any,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float,
) -> Any:
    layer = _layer_for_index(model, layer_idx)
    steer_vec = vector.detach().clone()

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        steer = steer_vec.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        new_hidden = hidden - alpha * steer
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _v1_topics() -> List[str]:
    labels: List[str] = []
    for cat_name, start, end in V1_CATEGORIES:
        labels.extend([cat_name] * (end - start))
    return labels


def _v2_topics() -> List[str]:
    categories = [
        "Tiananmen",
        "Tibet",
        "Xinjiang",
        "Xi/CCP governance",
        "Hong Kong",
        "COVID origins",
        "Taiwan",
        "Falun Gong",
        "Cultural Revolution / Mao",
        "Great Firewall",
        "South China Sea",
        "Internal CCP politics",
        "One-child policy",
        "Surveillance state",
        "Labor rights",
    ]
    labels: List[str] = []
    for cat_name in categories:
        labels.extend([cat_name] * 8)
    return labels


def resolve_prompt_pairs(corpus: str, limit_pairs: Optional[int]) -> List[PromptPair]:
    if corpus == "v1":
        topics = _v1_topics()
        pairs = [
            PromptPair(
                pair_idx=i,
                topic=topics[i],
                intensity="n/a",
                ccp_prompt=POLITICAL_REFUSED_PROMPTS[i],
                control_prompt=POLITICAL_ANSWERED_PROMPTS[i],
            )
            for i in range(len(POLITICAL_REFUSED_PROMPTS))
        ]
    elif corpus == "v2":
        topics = _v2_topics()
        pairs = [
            PromptPair(
                pair_idx=i,
                topic=topics[i],
                intensity="n/a",
                ccp_prompt=POLITICAL_REFUSED_PROMPTS_V2[i],
                control_prompt=POLITICAL_ANSWERED_PROMPTS_V2[i],
            )
            for i in range(len(POLITICAL_REFUSED_PROMPTS_V2))
        ]
    elif corpus == "adversarial":
        pairs = [
            PromptPair(
                pair_idx=i,
                topic=TOPIC_LABELS[i],
                intensity=INTENSITY_LABELS[i],
                ccp_prompt=ADVERSARIAL_CCP[i],
                control_prompt=ADVERSARIAL_PARALLEL[i],
            )
            for i in range(len(ADVERSARIAL_CCP))
        ]
    elif corpus == "safety_v1":
        pairs = [
            PromptPair(
                pair_idx=i,
                topic="safety",
                intensity="n/a",
                ccp_prompt=SAFETY_HARMFUL_PROMPTS[i],
                control_prompt=SAFETY_BENIGN_PROMPTS[i],
            )
            for i in range(len(SAFETY_HARMFUL_PROMPTS))
        ]
    elif corpus == "safety_v2":
        pairs = [
            PromptPair(
                pair_idx=i,
                topic="safety",
                intensity="n/a",
                ccp_prompt=SAFETY_HARMFUL_PROMPTS_V2[i],
                control_prompt=SAFETY_BENIGN_PROMPTS_V2[i],
            )
            for i in range(len(SAFETY_HARMFUL_PROMPTS_V2))
        ]
    elif corpus == "safety_v3":
        from safety_prompts_v3 import SAFETY_HARMFUL_V3, SAFETY_BENIGN_V3
        pairs = [
            PromptPair(
                pair_idx=i,
                topic="safety",
                intensity="n/a",
                ccp_prompt=SAFETY_HARMFUL_V3[i],
                control_prompt=SAFETY_BENIGN_V3[i],
            )
            for i in range(len(SAFETY_HARMFUL_V3))
        ]
    else:
        raise ValueError(f"Unsupported corpus: {corpus}")

    return pairs[:limit_pairs] if limit_pairs is not None else pairs


def _getattr_dotted(obj: Any, dotted_name: str) -> Optional[Any]:
    cur = obj
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_final_norm(model: Any) -> torch.nn.Module:
    candidates = [
        "model.norm",
        "model.model.norm",
        "transformer.ln_f",
        "transformer.final_layernorm",
        "model.final_layernorm",
        "model.model.final_layernorm",
        "language_model.model.norm",
        "language_model.norm",
        "model.language_model.norm",
        "model.model.language_model.norm",
        "language_model.transformer.ln_f",
        "model.transformer.ln_f",
        "transformer.encoder.final_layernorm",
        "model.encoder.final_layernorm",
    ]
    for name in candidates:
        module = _getattr_dotted(model, name)
        if isinstance(module, torch.nn.Module):
            return module
    return IdentityModule()


def resolve_output_head(model: Any) -> torch.nn.Module:
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if isinstance(head, torch.nn.Module):
            return head
    for name in ("lm_head", "output_layer", "embed_out"):
        head = _getattr_dotted(model, name)
        if isinstance(head, torch.nn.Module):
            return head
    raise AttributeError("Could not resolve output embedding / LM head")


def load_ablation_spec(path: Path) -> AblationSpec:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a dict checkpoint")

    if "best" in obj and "bank" in obj:
        best = obj["best"]
        bank = obj["bank"]
        candidate_idx = int(best["candidate_idx"])
        entry = bank[candidate_idx]
        return AblationSpec(
            layer_idx=int(best["layer_idx"]),
            alpha=float(best["alpha"]),
            vector=entry["v_clean"].detach().cpu().float(),
            source_path=str(path),
            source_kind="bank_best",
        )

    if "ridge_candidates" in obj:
        candidates = obj["ridge_candidates"]
        if not candidates:
            raise ValueError(f"{path} has no ridge_candidates")
        entry = sorted(candidates, key=lambda row: int(row["layer_idx"]))[0]
        return AblationSpec(
            layer_idx=int(entry["layer_idx"]),
            alpha=1.0,
            vector=entry["v_clean"].detach().cpu().float(),
            source_path=str(path),
            source_kind="ridge_candidate_first",
        )

    raise ValueError(f"Unsupported ablation checkpoint format: {path}")


def build_refusal_bundle(tokenizer: Any, refusal_strings: Sequence[str]) -> Dict[str, Any]:
    token_ids: List[int] = []
    variants_by_id: Dict[int, List[str]] = {}
    for raw in refusal_strings:
        for variant in (raw, f" {raw}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            token_id = int(ids[0])
            token_ids.append(token_id)
            variants_by_id.setdefault(token_id, []).append(variant)
    unique_ids = sorted(set(token_ids))
    return {
        "token_ids": unique_ids,
        "decoded": {tid: tokenizer.decode([tid], skip_special_tokens=False) for tid in unique_ids},
        "variants_by_token_id": variants_by_id,
        "source_strings": list(refusal_strings),
    }


def _run_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    ablation_spec: Optional[AblationSpec],
) -> torch.Tensor:
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    gen_cfg = _default_generate_kwargs(tokenizer)
    gen_cfg["max_new_tokens"] = max_new_tokens

    context = (
        temporary_ablation(
            model=model,
            layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None
        else nullcontext()
    )

    with context:
        with torch.no_grad():
            out = model.generate(**encoded, **gen_cfg)
    return out[0]


def find_first_meaningful_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    ablation_spec: Optional[AblationSpec],
    max_new_tokens: int,
) -> DecisionPoint:
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    input_len = encoded["input_ids"].shape[-1]
    generated = _run_generate(model, tokenizer, prompt, max_new_tokens, ablation_spec)
    new_ids = generated[input_len:].tolist()
    if not new_ids:
        raise RuntimeError("Model.generate produced no new tokens")

    special_ids = set()
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None and isinstance(value, int):
            special_ids.add(int(value))

    for idx, token_id in enumerate(new_ids):
        if int(token_id) in special_ids:
            continue
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if decoded.strip():
            return DecisionPoint(
                meaningful_token_id=int(token_id),
                prefix_token_ids=tuple(int(tok) for tok in new_ids[:idx]),
                generated_token_ids=tuple(int(tok) for tok in new_ids),
            )
    return DecisionPoint(
        meaningful_token_id=int(new_ids[0]),
        prefix_token_ids=tuple(),
        generated_token_ids=tuple(int(tok) for tok in new_ids),
    )


class nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def build_answer_bundle(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[PromptPair],
    ablation_spec: Optional[AblationSpec],
    max_new_tokens: int,
) -> Tuple[Dict[int, DecisionPoint], List[int], Dict[int, str]]:
    by_pair: Dict[int, DecisionPoint] = {}
    all_ids: List[int] = []
    decoded: Dict[int, str] = {}

    print(f"[1/4] Building answer bundle from {len(pairs)} control prompts... ({_timestamp()})")
    for idx, pair in enumerate(pairs, start=1):
        decision = find_first_meaningful_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=pair.control_prompt,
            ablation_spec=ablation_spec,
            max_new_tokens=max_new_tokens,
        )
        token_id = decision.meaningful_token_id
        by_pair[pair.pair_idx] = decision
        all_ids.append(token_id)
        decoded[token_id] = tokenizer.decode([token_id], skip_special_tokens=False)
        print(
            f"    answer token {idx}/{len(pairs)}: pair={pair.pair_idx} "
            f"id={token_id} text={decoded[token_id]!r}",
            end="\r",
            flush=True,
        )
    print(f"    answer bundle: done ({len(pairs)} pairs)          ")
    return by_pair, sorted(set(all_ids)), decoded


def _normalize_for_logits(final_norm: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0).unsqueeze(0)
    elif hidden.ndim == 2:
        hidden = hidden.unsqueeze(1)
    normed = final_norm(hidden)
    if normed.ndim == 3:
        return normed[:, 0, :]
    return normed


def forward_last_token_hidden_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    ablation_spec: Optional[AblationSpec],
) -> Tuple[List[torch.Tensor], int]:
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)
    attention_mask = encoded.get("attention_mask")
    last_idx = encoded["input_ids"].shape[-1] - 1 if attention_mask is None else int(attention_mask[0].sum().item()) - 1

    context = (
        temporary_ablation(
            model=model,
            layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None
        else nullcontext()
    )

    with context:
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = list(out.hidden_states[1:])
    return hidden_states, last_idx


def forward_context_hidden_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prefix_token_ids: Sequence[int],
    ablation_spec: Optional[AblationSpec],
) -> Tuple[List[torch.Tensor], int]:
    dev = _resolve_device(model)
    encoded = _tokenize(tokenizer, prompt, max_length=256, padding=False, chat_template=True)
    encoded = _to_device(encoded, dev)

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if prefix_token_ids:
        prefix = torch.tensor([list(prefix_token_ids)], device=input_ids.device, dtype=input_ids.dtype)
        prefix_mask = torch.ones_like(prefix, device=attention_mask.device, dtype=attention_mask.dtype)
        input_ids = torch.cat([input_ids, prefix], dim=-1)
        attention_mask = torch.cat([attention_mask, prefix_mask], dim=-1)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in encoded:
        token_type_ids = encoded["token_type_ids"]
        if prefix_token_ids:
            prefix_tti = torch.zeros(
                (1, len(prefix_token_ids)),
                device=token_type_ids.device,
                dtype=token_type_ids.dtype,
            )
            token_type_ids = torch.cat([token_type_ids, prefix_tti], dim=-1)
        model_inputs["token_type_ids"] = token_type_ids

    last_idx = int(attention_mask[0].sum().item()) - 1

    context = (
        temporary_ablation(
            model=model,
            layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None
        else nullcontext()
    )

    with context:
        with torch.no_grad():
            out = model(**model_inputs, output_hidden_states=True, use_cache=False)

    hidden_states = list(out.hidden_states[1:])
    return hidden_states, last_idx


def compute_prompt_logits_by_layer(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    prompt: str,
    ablation_spec: Optional[AblationSpec],
    position_mode: str,
    decision_point: Optional[DecisionPoint],
) -> List[torch.Tensor]:
    """Return full logit vectors at each layer for the chosen position."""
    if position_mode == "last_prompt":
        layer_hidden_states, last_idx = forward_last_token_hidden_states(
            model=model, tokenizer=tokenizer, prompt=prompt, ablation_spec=ablation_spec,
        )
    elif position_mode == "first_meaningful":
        if decision_point is None:
            raise ValueError("decision_point is required for position_mode=first_meaningful")
        layer_hidden_states, last_idx = forward_context_hidden_states(
            model=model, tokenizer=tokenizer, prompt=prompt,
            prefix_token_ids=decision_point.prefix_token_ids, ablation_spec=ablation_spec,
        )
    else:
        raise ValueError(f"Unsupported position_mode: {position_mode}")
    logits_by_layer: List[torch.Tensor] = []
    for hidden in layer_hidden_states:
        token_hidden = hidden[0, last_idx, :]
        normed = _normalize_for_logits(final_norm, token_hidden)
        logits = lm_head(normed).squeeze(0).detach().float().cpu()
        logits_by_layer.append(logits)
    return logits_by_layer


def compute_pair_kl_records(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    ccp_prompt: str,
    control_prompt: str,
    ablation_spec: Optional[AblationSpec],
    position_mode: str,
    ccp_decision: Optional[DecisionPoint],
    control_decision: Optional[DecisionPoint],
) -> List[Dict[str, Any]]:
    """Compute per-layer KL(softmax(ccp) || softmax(control)) for a matched pair."""
    import torch.nn.functional as F

    ccp_logits = compute_prompt_logits_by_layer(
        model, tokenizer, final_norm, lm_head, ccp_prompt,
        ablation_spec, position_mode, ccp_decision,
    )
    ctrl_logits = compute_prompt_logits_by_layer(
        model, tokenizer, final_norm, lm_head, control_prompt,
        ablation_spec, position_mode, control_decision,
    )
    rows: List[Dict[str, Any]] = []
    for layer_idx, (cl, ctl) in enumerate(zip(ccp_logits, ctrl_logits)):
        log_p = F.log_softmax(cl, dim=-1)
        log_q = F.log_softmax(ctl, dim=-1)
        p = log_p.exp()
        kl_pq = float((p * (log_p - log_q)).sum().item())
        q = log_q.exp()
        kl_qp = float((q * (log_q - log_p)).sum().item())
        rows.append({
            "layer": int(layer_idx),
            "kl_ccp_ctrl": kl_pq,
            "kl_ctrl_ccp": kl_qp,
            "kl_symmetric": (kl_pq + kl_qp) / 2.0,
        })
    return rows


def compute_pair_teacher_forced_records(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    ccp_prompt: str,
    control_prompt: str,
    ablation_spec: Optional[AblationSpec],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """Teacher-forced log-prob of the control continuation under the CCP prompt.

    For each layer, ask: given the CCP prompt, how likely is the model to produce
    the same first few tokens it would produce for the matched control?  Returns
    per-layer mean negative log-prob of the control continuation.  A higher value
    means more divergence = more routing.
    """
    import torch.nn.functional as F

    # Step 1: generate the control continuation
    dev = _resolve_device(model)
    ctrl_gen = _run_generate(model, tokenizer, control_prompt, max_new_tokens, ablation_spec)
    ctrl_encoded = _tokenize(tokenizer, control_prompt, max_length=256, padding=False, chat_template=True)
    ctrl_input_len = ctrl_encoded["input_ids"].shape[-1]
    ctrl_continuation_ids = ctrl_gen[ctrl_input_len:].tolist()
    # Take first few non-special content tokens (up to max_new_tokens)
    content_ids: List[int] = []
    special_ids = set()
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None and isinstance(value, int):
            special_ids.add(int(value))
    for tid in ctrl_continuation_ids:
        if int(tid) not in special_ids:
            content_ids.append(int(tid))
        if len(content_ids) >= max_new_tokens:
            break
    if not content_ids:
        content_ids = [int(ctrl_continuation_ids[0])] if ctrl_continuation_ids else [0]

    # Step 2: feed CCP prompt + control continuation prefix, get hidden states
    # at each position corresponding to a control continuation token
    ccp_encoded = _tokenize(tokenizer, ccp_prompt, max_length=256, padding=False, chat_template=True)
    ccp_input_ids = ccp_encoded["input_ids"]
    ccp_mask = ccp_encoded.get("attention_mask")
    if ccp_mask is None:
        ccp_mask = torch.ones_like(ccp_input_ids)

    # Append control continuation tokens to CCP prompt
    cont_tensor = torch.tensor([content_ids], dtype=ccp_input_ids.dtype)
    full_ids = torch.cat([ccp_input_ids, cont_tensor], dim=-1).to(dev)
    full_mask = torch.cat([ccp_mask, torch.ones_like(cont_tensor)], dim=-1).to(dev)

    context = (
        temporary_ablation(
            model=model,
            layer_idx=ablation_spec.layer_idx,
            vector=ablation_spec.vector.to(dev, dtype=torch.float32),
            alpha=ablation_spec.alpha,
        )
        if ablation_spec is not None
        else nullcontext()
    )
    with context:
        with torch.no_grad():
            out = model(
                input_ids=full_ids, attention_mask=full_mask,
                output_hidden_states=True, use_cache=False,
            )

    ccp_prompt_len = int(ccp_mask[0].sum().item())
    n_content = len(content_ids)

    # Step 3: for each layer, compute mean NLL of control continuation
    rows: List[Dict[str, Any]] = []
    for layer_idx, hidden in enumerate(out.hidden_states[1:]):
        total_nll = 0.0
        for pos_offset in range(n_content):
            # Position in full sequence predicting content_ids[pos_offset]
            pred_pos = ccp_prompt_len - 1 + pos_offset
            token_hidden = hidden[0, pred_pos, :]
            normed = _normalize_for_logits(final_norm, token_hidden)
            logits = lm_head(normed).squeeze(0).detach().float().cpu()
            log_probs = F.log_softmax(logits, dim=-1)
            target_id = content_ids[pos_offset]
            total_nll -= float(log_probs[target_id].item())
        mean_nll = total_nll / n_content
        rows.append({
            "layer": int(layer_idx),
            "mean_nll_ctrl_continuation": mean_nll,
            "n_content_tokens": n_content,
        })
    return rows


def compute_prompt_layer_records(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    prompt: str,
    refusal_ids: Sequence[int],
    global_answer_ids: Sequence[int],
    pair_answer_id: int,
    ablation_spec: Optional[AblationSpec],
    position_mode: str,
    decision_point: Optional[DecisionPoint],
) -> List[Dict[str, Any]]:
    if position_mode == "last_prompt":
        layer_hidden_states, last_idx = forward_last_token_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            ablation_spec=ablation_spec,
        )
    elif position_mode == "first_meaningful":
        if decision_point is None:
            raise ValueError("decision_point is required for position_mode=first_meaningful")
        layer_hidden_states, last_idx = forward_context_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prefix_token_ids=decision_point.prefix_token_ids,
            ablation_spec=ablation_spec,
        )
    else:
        raise ValueError(f"Unsupported position_mode: {position_mode}")
    rows: List[Dict[str, Any]] = []
    for layer_idx, hidden in enumerate(layer_hidden_states):
        token_hidden = hidden[0, last_idx, :]
        normed = _normalize_for_logits(final_norm, token_hidden)
        logits = lm_head(normed).squeeze(0).detach().float().cpu()
        refusal_mean = float(logits[list(refusal_ids)].mean().item())
        answer_mean = float(logits[list(global_answer_ids)].mean().item())
        pair_answer_logit = float(logits[int(pair_answer_id)].item())
        topk = torch.topk(logits, k=5)
        top_ids = [int(t) for t in topk.indices.tolist()]
        top_text = [tokenizer.decode([tid], skip_special_tokens=False) for tid in top_ids]
        rows.append(
            {
                "layer": int(layer_idx),
                "refusal_logit_mean": refusal_mean,
                "answer_logit_mean": answer_mean,
                "pair_answer_logit": pair_answer_logit,
                "global_logit_diff": refusal_mean - answer_mean,
                "pair_logit_diff": refusal_mean - pair_answer_logit,
                "top_token_id": top_ids[0],
                "top_token_text": top_text[0],
                "top5_token_ids": top_ids,
                "top5_token_text": top_text,
            }
        )
    return rows


def bootstrap_summary(values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = values.size
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        means.append(float(values[sample].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_kl_trajectory(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    pairs: List[PromptPair],
    ablation_spec: Optional[AblationSpec],
    args: Any,
    run_dir: Path,
) -> None:
    """KL divergence trajectory between CCP and matched-control distributions."""
    ccp_decisions: Dict[int, DecisionPoint] = {}
    ctrl_decisions: Dict[int, DecisionPoint] = {}

    if args.position == "first_meaningful":
        print(f"[1/4] Resolving decision positions... ({_timestamp()})")
        for idx, pair in enumerate(pairs, start=1):
            ccp_decisions[pair.pair_idx] = find_first_meaningful_generation(
                model=model, tokenizer=tokenizer, prompt=pair.ccp_prompt,
                ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
            )
            ctrl_decisions[pair.pair_idx] = find_first_meaningful_generation(
                model=model, tokenizer=tokenizer, prompt=pair.control_prompt,
                ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
            )
            print(f"    decision positions {idx}/{len(pairs)}", end="\r", flush=True)
        print(f"    decision positions: done ({len(pairs)} pairs)          ")

    print(f"[2/4] Computing layerwise KL trajectories... ({_timestamp()})")
    pair_rows: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        kl_rows = compute_pair_kl_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm, lm_head=lm_head,
            ccp_prompt=pair.ccp_prompt, control_prompt=pair.control_prompt,
            ablation_spec=ablation_spec, position_mode=args.position,
            ccp_decision=ccp_decisions.get(pair.pair_idx),
            control_decision=ctrl_decisions.get(pair.pair_idx),
        )
        for row in kl_rows:
            row["pair_idx"] = pair.pair_idx
            row["topic"] = pair.topic
            row["intensity"] = pair.intensity
            pair_rows.append(row)
        print(f"    pair {idx}/{len(pairs)} complete: topic={pair.topic}", end="\r", flush=True)
    print(f"    KL trajectories: done ({len(pairs)} pairs)          ")

    print(f"[3/4] Aggregating summaries... ({_timestamp()})")
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    by_layer_topic: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
        by_layer_topic.setdefault((int(row["layer"]), str(row["topic"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        sym_vals = np.asarray([float(r["kl_symmetric"]) for r in rows], dtype=np.float64)
        fwd_vals = np.asarray([float(r["kl_ccp_ctrl"]) for r in rows], dtype=np.float64)
        rev_vals = np.asarray([float(r["kl_ctrl_ccp"]) for r in rows], dtype=np.float64)
        ci_low, ci_high = bootstrap_summary(sym_vals, args.bootstrap, args.seed + layer)
        summary_rows.append({
            "layer": layer,
            "n_pairs": len(rows),
            "mean_kl_symmetric": float(sym_vals.mean()),
            "kl_sym_ci_low": ci_low,
            "kl_sym_ci_high": ci_high,
            "mean_kl_ccp_ctrl": float(fwd_vals.mean()),
            "mean_kl_ctrl_ccp": float(rev_vals.mean()),
        })

    topic_rows: List[Dict[str, Any]] = []
    for (layer, topic), rows in sorted(by_layer_topic.items()):
        sym_vals = np.asarray([float(r["kl_symmetric"]) for r in rows], dtype=np.float64)
        topic_rows.append({
            "layer": layer,
            "topic": topic,
            "n_pairs": len(rows),
            "mean_kl_symmetric": float(sym_vals.mean()),
        })

    metadata = {
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "position": args.position,
        "metric": "kl",
        "run_dir": str(run_dir),
        "n_pairs": len(pairs),
        "ablation": None if ablation_spec is None else {
            "layer_idx": ablation_spec.layer_idx,
            "alpha": ablation_spec.alpha,
            "source_path": ablation_spec.source_path,
            "source_kind": ablation_spec.source_kind,
        },
    }

    print(f"[4/4] Writing outputs... ({_timestamp()})")
    write_csv(run_dir / "kl_pairwise_by_layer.csv", pair_rows)
    write_csv(run_dir / "kl_trajectory_summary.csv", summary_rows)
    write_csv(run_dir / "kl_trajectory_by_topic.csv", topic_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print("Done.")
    print(f"  Summary: {run_dir / 'kl_trajectory_summary.csv'}")
    print(f"  Pairwise: {run_dir / 'kl_pairwise_by_layer.csv'}")
    print(f"  By topic: {run_dir / 'kl_trajectory_by_topic.csv'}")


def _run_teacher_forced_trajectory(
    model: Any,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    pairs: List[PromptPair],
    ablation_spec: Optional[AblationSpec],
    args: Any,
    run_dir: Path,
) -> None:
    """Teacher-forced NLL trajectory: how unlikely is the control continuation under the CCP prompt?

    For each pair, generates the control prompt's continuation, then teacher-forces
    those tokens through the CCP prompt's forward pass.  Per-layer mean NLL of
    the control continuation measures how much the CCP context diverges from the
    control answer trajectory.  The pairwise delta (CCP NLL minus control NLL)
    isolates the routing effect.
    """

    print(f"[1/4] Computing teacher-forced trajectories... ({_timestamp()})")
    pair_rows: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        # NLL of control continuation under CCP prompt
        ccp_rows = compute_pair_teacher_forced_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm, lm_head=lm_head,
            ccp_prompt=pair.ccp_prompt, control_prompt=pair.control_prompt,
            ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
        )
        # NLL of control continuation under control prompt (baseline)
        ctrl_rows = compute_pair_teacher_forced_records(
            model=model, tokenizer=tokenizer, final_norm=final_norm, lm_head=lm_head,
            ccp_prompt=pair.control_prompt, control_prompt=pair.control_prompt,
            ablation_spec=ablation_spec, max_new_tokens=args.answer_max_new_tokens,
        )
        for ccp_row, ctrl_row in zip(ccp_rows, ctrl_rows):
            pair_rows.append({
                "pair_idx": pair.pair_idx,
                "topic": pair.topic,
                "intensity": pair.intensity,
                "layer": ccp_row["layer"],
                "n_content_tokens": ccp_row["n_content_tokens"],
                "ccp_nll": ccp_row["mean_nll_ctrl_continuation"],
                "ctrl_nll": ctrl_row["mean_nll_ctrl_continuation"],
                "delta_nll": ccp_row["mean_nll_ctrl_continuation"] - ctrl_row["mean_nll_ctrl_continuation"],
            })
        print(f"    pair {idx}/{len(pairs)} complete: topic={pair.topic}", end="\r", flush=True)
    print(f"    teacher-forced: done ({len(pairs)} pairs)          ")

    print(f"[2/4] Aggregating summaries... ({_timestamp()})")
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    by_layer_topic: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
        by_layer_topic.setdefault((int(row["layer"]), str(row["topic"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        delta_vals = np.asarray([float(r["delta_nll"]) for r in rows], dtype=np.float64)
        ccp_vals = np.asarray([float(r["ccp_nll"]) for r in rows], dtype=np.float64)
        ctrl_vals = np.asarray([float(r["ctrl_nll"]) for r in rows], dtype=np.float64)
        ci_low, ci_high = bootstrap_summary(delta_vals, args.bootstrap, args.seed + layer)
        summary_rows.append({
            "layer": layer,
            "n_pairs": len(rows),
            "mean_delta_nll": float(delta_vals.mean()),
            "delta_nll_ci_low": ci_low,
            "delta_nll_ci_high": ci_high,
            "mean_ccp_nll": float(ccp_vals.mean()),
            "mean_ctrl_nll": float(ctrl_vals.mean()),
        })

    topic_rows: List[Dict[str, Any]] = []
    for (layer, topic), rows in sorted(by_layer_topic.items()):
        delta_vals = np.asarray([float(r["delta_nll"]) for r in rows], dtype=np.float64)
        topic_rows.append({
            "layer": layer,
            "topic": topic,
            "n_pairs": len(rows),
            "mean_delta_nll": float(delta_vals.mean()),
        })

    metadata = {
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "position": "teacher_forced",
        "metric": "teacher_forced",
        "run_dir": str(run_dir),
        "n_pairs": len(pairs),
        "answer_max_new_tokens": args.answer_max_new_tokens,
        "ablation": None if ablation_spec is None else {
            "layer_idx": ablation_spec.layer_idx,
            "alpha": ablation_spec.alpha,
            "source_path": ablation_spec.source_path,
            "source_kind": ablation_spec.source_kind,
        },
    }

    print(f"[3/4] Writing outputs... ({_timestamp()})")
    write_csv(run_dir / "tf_pairwise_by_layer.csv", pair_rows)
    write_csv(run_dir / "tf_trajectory_summary.csv", summary_rows)
    write_csv(run_dir / "tf_trajectory_by_topic.csv", topic_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print("Done.")
    print(f"  Summary: {run_dir / 'tf_trajectory_summary.csv'}")
    print(f"  Pairwise: {run_dir / 'tf_pairwise_by_layer.csv'}")
    print(f"  By topic: {run_dir / 'tf_trajectory_by_topic.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus", choices=["v1", "adversarial"], default="v1")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--refusal-strings", nargs="+", default=list(DEFAULT_REFUSAL_STRINGS))
    parser.add_argument("--answer-max-new-tokens", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--position",
        choices=["first_meaningful", "last_prompt"],
        default="first_meaningful",
        help="Token position to analyze. 'first_meaningful' uses the context immediately before the first semantically meaningful generated token; 'last_prompt' uses the last prompt token as a precursor view.",
    )
    parser.add_argument(
        "--metric",
        choices=["logit_diff", "kl", "teacher_forced"],
        default="logit_diff",
        help="Trajectory metric. 'logit_diff' computes pairwise refusal-vs-answer logit difference (default). "
        "'kl' computes per-layer KL divergence between CCP and control next-token distributions. "
        "'teacher_forced' computes mean NLL of the control continuation under the CCP prompt at each layer.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit output directory. If omitted, allocate a run ID under runs/.",
    )
    parser.add_argument(
        "--ablation-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint with a reusable clean direction and selected layer/alpha.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = resolve_prompt_pairs(args.corpus, args.limit_pairs)
    base_dir = Path("runs")
    run_dir = Path(args.run_dir) if args.run_dir else _next_run_dir(base_dir, args.model, args.corpus, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    ablation_spec = load_ablation_spec(Path(args.ablation_checkpoint)) if args.ablation_checkpoint else None

    print("=" * 80)
    print("Routing Logit Trajectory")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Pairs: {len(pairs)}")
    print(f"Position: {args.position}")
    print(f"Metric: {args.metric}")
    print(f"Run dir: {run_dir}")
    if ablation_spec is not None:
        print(
            f"Ablation: layer={ablation_spec.layer_idx} alpha={ablation_spec.alpha:.4f} "
            f"source={ablation_spec.source_path}"
        )
    print("=" * 80)

    print(f"[0/4] Loading model/tokenizer... ({_timestamp()})")
    model, tokenizer = load_model_and_tokenizer(args.model)
    final_norm = resolve_final_norm(model)
    lm_head = resolve_output_head(model)

    if args.metric == "kl":
        _run_kl_trajectory(model, tokenizer, final_norm, lm_head, pairs, ablation_spec, args, run_dir)
        return

    if args.metric == "teacher_forced":
        _run_teacher_forced_trajectory(model, tokenizer, final_norm, lm_head, pairs, ablation_spec, args, run_dir)
        return

    refusal_bundle = build_refusal_bundle(tokenizer, args.refusal_strings)
    refusal_ids = refusal_bundle["token_ids"]
    if not refusal_ids:
        raise ValueError("Refusal bundle is empty after tokenization")

    answer_decisions_by_pair, global_answer_ids, decoded_answer_ids = build_answer_bundle(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        ablation_spec=ablation_spec,
        max_new_tokens=args.answer_max_new_tokens,
    )

    ccp_decisions_by_pair: Dict[int, DecisionPoint] = {}
    if args.position == "first_meaningful":
        print(f"[1.5/4] Resolving CCP decision positions... ({_timestamp()})")
        for idx, pair in enumerate(pairs, start=1):
            decision = find_first_meaningful_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=pair.ccp_prompt,
                ablation_spec=ablation_spec,
                max_new_tokens=args.answer_max_new_tokens,
            )
            ccp_decisions_by_pair[pair.pair_idx] = decision
            decoded = tokenizer.decode([decision.meaningful_token_id], skip_special_tokens=False)
            print(
                f"    ccp token {idx}/{len(pairs)}: pair={pair.pair_idx} "
                f"id={decision.meaningful_token_id} text={decoded!r}",
                end="\r",
                flush=True,
            )
        print(f"    ccp positions: done ({len(pairs)} pairs)          ")

    print(f"[2/4] Computing layerwise trajectories... ({_timestamp()})")
    pair_rows: List[Dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        pair_answer_id = answer_decisions_by_pair[pair.pair_idx].meaningful_token_id
        ccp_rows = compute_prompt_layer_records(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            lm_head=lm_head,
            prompt=pair.ccp_prompt,
            refusal_ids=refusal_ids,
            global_answer_ids=global_answer_ids,
            pair_answer_id=pair_answer_id,
            ablation_spec=ablation_spec,
            position_mode=args.position,
            decision_point=ccp_decisions_by_pair.get(pair.pair_idx),
        )
        ctrl_rows = compute_prompt_layer_records(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            lm_head=lm_head,
            prompt=pair.control_prompt,
            refusal_ids=refusal_ids,
            global_answer_ids=global_answer_ids,
            pair_answer_id=pair_answer_id,
            ablation_spec=ablation_spec,
            position_mode=args.position,
            decision_point=answer_decisions_by_pair.get(pair.pair_idx),
        )
        for ccp_row, ctrl_row in zip(ccp_rows, ctrl_rows):
            pair_rows.append(
                {
                    "pair_idx": pair.pair_idx,
                    "topic": pair.topic,
                    "intensity": pair.intensity,
                    "layer": ccp_row["layer"],
                    "pair_answer_token_id": pair_answer_id,
                    "pair_answer_token_text": decoded_answer_ids[pair_answer_id],
                    "ccp_pair_logit_diff": ccp_row["pair_logit_diff"],
                    "control_pair_logit_diff": ctrl_row["pair_logit_diff"],
                    "delta_pair_logit_diff": ccp_row["pair_logit_diff"] - ctrl_row["pair_logit_diff"],
                    "ccp_global_logit_diff": ccp_row["global_logit_diff"],
                    "control_global_logit_diff": ctrl_row["global_logit_diff"],
                    "delta_global_logit_diff": ccp_row["global_logit_diff"] - ctrl_row["global_logit_diff"],
                    "ccp_top_token_id": ccp_row["top_token_id"],
                    "ccp_top_token_text": ccp_row["top_token_text"],
                    "control_top_token_id": ctrl_row["top_token_id"],
                    "control_top_token_text": ctrl_row["top_token_text"],
                }
            )
        print(f"    pair {idx}/{len(pairs)} complete: topic={pair.topic}", end="\r", flush=True)
    print(f"    trajectories: done ({len(pairs)} pairs)          ")

    print(f"[3/4] Aggregating summaries... ({_timestamp()})")
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    by_layer_topic: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
        by_layer_topic.setdefault((int(row["layer"]), str(row["topic"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        delta_vals = np.asarray([float(r["delta_pair_logit_diff"]) for r in rows], dtype=np.float64)
        ccp_vals = np.asarray([float(r["ccp_pair_logit_diff"]) for r in rows], dtype=np.float64)
        ctrl_vals = np.asarray([float(r["control_pair_logit_diff"]) for r in rows], dtype=np.float64)
        ci_low, ci_high = bootstrap_summary(delta_vals, args.bootstrap, args.seed + layer)
        summary_rows.append(
            {
                "layer": layer,
                "n_pairs": len(rows),
                "mean_delta_pair_logit_diff": float(delta_vals.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_ccp_pair_logit_diff": float(ccp_vals.mean()),
                "mean_control_pair_logit_diff": float(ctrl_vals.mean()),
            }
        )

    topic_rows: List[Dict[str, Any]] = []
    for (layer, topic), rows in sorted(by_layer_topic.items()):
        delta_vals = np.asarray([float(r["delta_pair_logit_diff"]) for r in rows], dtype=np.float64)
        topic_rows.append(
            {
                "layer": layer,
                "topic": topic,
                "n_pairs": len(rows),
                "mean_delta_pair_logit_diff": float(delta_vals.mean()),
            }
        )

    metadata = {
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "position": args.position,
        "metric": "logit_diff",
        "run_dir": str(run_dir),
        "n_pairs": len(pairs),
        "refusal_bundle": refusal_bundle,
        "global_answer_token_ids": global_answer_ids,
        "global_answer_token_text": {tid: decoded_answer_ids[tid] for tid in global_answer_ids},
        "ablation": None
        if ablation_spec is None
        else {
            "layer_idx": ablation_spec.layer_idx,
            "alpha": ablation_spec.alpha,
            "source_path": ablation_spec.source_path,
            "source_kind": ablation_spec.source_kind,
        },
    }

    print(f"[4/4] Writing outputs... ({_timestamp()})")
    write_csv(run_dir / "pairwise_by_layer.csv", pair_rows)
    write_csv(run_dir / "trajectory_summary.csv", summary_rows)
    write_csv(run_dir / "trajectory_by_topic.csv", topic_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"  Summary: {run_dir / 'trajectory_summary.csv'}")
    print(f"  Pairwise: {run_dir / 'pairwise_by_layer.csv'}")
    print(f"  By topic: {run_dir / 'trajectory_by_topic.csv'}")


if __name__ == "__main__":
    main()
