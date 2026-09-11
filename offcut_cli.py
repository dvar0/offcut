#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Offcut's generation CLI using Comfy's low-level inference runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_COMFY_ROOT = APP_ROOT.parent / "ComfyUI"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "offcut" / "settings.json"
SETTINGS_LOCK = threading.RLock()

DEFAULT_SYSTEM_PROMPT = """You are an expert prompt engineer for Krea 2 image generation.

Rewrite the user's image prompt into one detailed, visually grounded image prompt.
Krea 2 rewards natural-language description over keyword lists. Scale the length to how much the
request actually specifies: a rich request supports 80-150 words of flowing prose or comma-separated
clauses, while a deliberately minimal or abstract one should stay short rather than be padded with
invented detail. Krea 2's own reference prompts run from 10 to 180 words.

Cover the axes that serve the image, in whatever order reads naturally: subject and its attributes,
pose or action, wardrobe and materials, setting and background, lighting, color palette,
medium or aesthetic, and framing (shot distance, angle, lens, depth of field).

Rules:
- Output only the final prompt. No markdown, labels, notes, or visible reasoning. Plan silently.
- Keep the prompt as one paragraph.
- Preserve the user's subject, action, setting, composition, and requested medium. Never pivot to a
  different medium to make the image easier.
- Do not invent new main subjects, characters, animals, props, logos, or text unless clearly implied.
- Do not over-specify clothing, colors, or materials that the input does not support.
- If the user wants visible text, reproduce the exact words wrapped in "quotes".
- If the user's prompt is already long and detailed, lightly polish and finalize it and preserve their
  phrasing rather than expanding it further.
- Treat depictions of people with dignity. Assume clothing covers genitals and intimate anatomy.
- If a LoRA trigger/style is provided, keep that trigger phrase verbatim at the beginning of the prompt and preserve that style direction.
- Do not replace or dilute the LoRA style with a conflicting style, artist, medium, or genre.
"""

# The one frame every style and LoRA in the library is showcased with, so the covers form a
# comparison rather than a collection of unrelated pictures. What earns a place here is not detail
# for its own sake: it is content at three spatial frequencies at once, which is what actually
# separates two styles side by side. One large lit plane (the carved stone) carries surface texture,
# one large empty dark mass (the void above) is the only place grain, dither and brushwork are
# visible at all, and a field of many small repeated lights (the moths) forces a decision about
# mark-making at minimum size — the same field renders as flat shapes, soft halos, or single pixels
# depending on the adapter. The armor is an ornamented hard specular surface, the pool tests
# reflection, and the up-lit face tests skin and falloff together.
#
# Two things are deliberate rather than incidental. No medium word appears anywhere, so the style
# text or LoRA trigger prepended in front of this decides the medium instead of competing with it.
# And the subject is described with adult build, wardrobe and expression, because scale words like
# "a small figure" are read as a small person rather than a small area of the frame and reliably
# produce a child.
DEFAULT_COVER_PROMPT = (
    "A tall woman in ornate blackened iron armor kneels on one knee at the edge of a still black "
    "pool, one gauntlet pressed flat to the water, her long white hair loose and lifting. Dozens of "
    "small luminous moths rise from the surface around her, trailing bright cyan streaks. Behind "
    "her a broad wall of pale carved stone catches their light across its near face and falls away "
    "into empty black above. Her face is turned up toward them, lit from below, calm and hard. "
    "Cracked wet flagstone underfoot holds the reflections. Deep black and cobalt with pale stone "
    "and one warm gold accent at her collar. Square composition, full figure, low angle."
)

# A cover is a showcase, not a good-taste sample: it is meant to make the adapter's hand as obvious
# as it can be, which is why the strength here is well above the roughly 1.0 normally recommended
# for real work. Nothing else reads this value, so overdriving it cannot leak into a
# normal generation.
DEFAULT_COVER_LORA_STRENGTH = 2.0

DEFAULT_SETTINGS: dict[str, Any] = {
    "comfy_root": str(DEFAULT_COMFY_ROOT),
    "model_dir": str(APP_ROOT / "models"),
    "output_dir": str(APP_ROOT / "outputs"),
    "text_encoder": str(DEFAULT_COMFY_ROOT / "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    "vae": str(DEFAULT_COMFY_ROOT / "models/vae/qwen_image_vae.safetensors"),
    "lora_dirs": [
        str(APP_ROOT / "models" / "loras"),
        str(DEFAULT_COMFY_ROOT / "models/loras"),
    ],
    "lora_metadata": {},
    "cover": {
        "prompt": DEFAULT_COVER_PROMPT,
        "seed": 20260904,
        "preset": "raw-int8-turbo-lora",
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "guidance": 0.0,
        "raw_portion": 8.0,
        "raw_steps": 52,
        "lora_strength": DEFAULT_COVER_LORA_STRENGTH,
    },
    "enhancer": {
        "enabled": False,
        "endpoint": "https://opencode.ai/zen/go/v1/chat/completions",
        "model": "glm-5.2",
        "api_key_env": "OPENCODE_GO_API_KEY",
        "temperature": 0.4,
        "max_tokens": 1024,
        "timeout_seconds": 120,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
}


@dataclass(frozen=True)
class Download:
    name: str
    relative_path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/Comfy-Org/Krea-2/resolve/main/{self.relative_path}"


DOWNLOADS = {
    "raw-int8": Download(
        "raw-int8",
        "diffusion_models/krea2_raw_int8_convrot.safetensors",
        13_492_686_496,
        "5585a4a38c4bcfb6fde2d480a4aa6edf7f665721ebde56d30662c35a45f5fa5c",
    ),
    "turbo-lora": Download(
        "turbo-lora",
        "loras/krea2_turbo_lora_rank_64_bf16.safetensors",
        469_423_778,
        "db8c5bae0a415d448da9d842111d6e51f7d32e47143a3118eb267e5c4773de87",
    ),
}


@dataclass(frozen=True)
class Preset:
    name: str
    checkpoint: str
    turbo: bool
    turbo_lora: bool = False
    default_steps: int = 8
    default_guidance: float = 0.0
    two_stage: bool = False


PRESETS = {
    "raw-int8-turbo-lora": Preset(
        "raw-int8-turbo-lora",
        "krea2_raw_int8_convrot.safetensors",
        turbo=True,
        turbo_lora=True,
    ),
    "raw-int8": Preset(
        "raw-int8",
        "krea2_raw_int8_convrot.safetensors",
        turbo=False,
        default_steps=52,
        default_guidance=3.5,
    ),
    "raw-int8-to-turbo": Preset(
        "raw-int8-to-turbo",
        "krea2_raw_int8_convrot.safetensors",
        turbo=False,
        turbo_lora=True,
        default_steps=12,
        default_guidance=3.0,
        two_stage=True,
    ),
}

DEFAULT_PRESET = "raw-int8-turbo-lora"
HYBRID_PRESET = "raw-int8-to-turbo"


def migrate_preset(name: str) -> str:
    """Translate active settings only. Historical image recipes keep their original route."""
    return DEFAULT_PRESET if name == "turbo-int8" else name


def hybrid_settings(preset: Preset, steps: int, raw_portion: Any = None, raw_steps: Any = None) -> dict[str, Any]:
    if not preset.two_stage:
        return {}
    portion = 8.0 if raw_portion in (None, "") else raw_portion
    density = 52 if raw_steps in (None, "") else raw_steps
    if isinstance(portion, bool) or not isinstance(portion, (int, float)) or not math.isfinite(portion) or not 0 < portion < 100:
        raise ValueError("Raw portion must be a finite percentage greater than 0 and less than 100; use Turbo or Raw for the endpoints")
    if isinstance(density, bool) or not isinstance(density, int) or not 2 <= density <= 100:
        raise ValueError("Raw schedule steps must be an integer between 2 and 100")
    if not 2 <= steps <= 100:
        raise ValueError("Turbo schedule steps must be between 2 and 100 for Raw → Turbo")
    return {"raw_portion": float(portion), "raw_steps": density}


def split_sigma_schedules(raw_sigmas: Any, turbo_sigmas: Any, raw_portion: float) -> tuple[Any, Any]:
    """Cut the raw schedule, then sigma-lock the nearest valid turbo continuation.

    Schedule densities are not executed step counts. The boundary stays nonzero in raw and
    is exactly the first sigma in turbo; only the second stage runs to zero.
    """
    end = max(1, min(len(raw_sigmas) - 2, round((len(raw_sigmas) - 1) * raw_portion / 100)))
    boundary = raw_sigmas[end]
    if not 0 < float(boundary) < float(turbo_sigmas[0]):
        raise ValueError("Raw handoff is outside the Turbo schedule")
    candidates = [i for i in range(1, len(turbo_sigmas) - 1) if float(turbo_sigmas[i - 1]) > float(boundary)]
    if not candidates:
        raise ValueError("Turbo schedule has no non-terminal handoff")
    start = min(candidates, key=lambda i: abs(float(turbo_sigmas[i]) - float(boundary)))
    raw = raw_sigmas[:end + 1].clone()
    turbo = turbo_sigmas[start:].clone()
    turbo[0] = boundary
    return raw, turbo


def hybrid_step_plan(width: int, height: int, raw_portion: float = 8.0, raw_steps: int = 52, steps: int = 12) -> dict[str, int]:
    """Preview Comfy's Flux/simple handoff without importing torch or loading the engine.

    Keep in step with web/sampling.js. Float32 at each tensor operation matters near the
    nearest-sigma boundary; Python's ordinary float arithmetic alone can pick a different step.
    """
    validate_dimensions(width, height)
    recipe = hybrid_settings(PRESETS[HYBRID_PRESET], steps, raw_portion, raw_steps)
    raw_steps, raw_portion = recipe["raw_steps"], recipe["raw_portion"]

    def f32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    def schedule(shift: float, density: int) -> list[float]:
        exp = f32(math.exp(shift))
        result = []
        for index in range(density):
            time = f32((10000 - int(index * (10000 / density))) / 10000)
            result.append(f32(exp / f32(exp + f32(f32(1 / time) - 1))))
        return result

    raw_count = max(1, min(raw_steps - 1, round(raw_steps * raw_portion / 100)))
    boundary = schedule(raw_sampling_shift(width, height), raw_steps)[raw_count]
    turbo = schedule(1.15, steps)
    start = min((i for i in range(1, steps) if turbo[i - 1] > boundary), key=lambda i: abs(turbo[i] - boundary))
    return {"raw": raw_count, "turbo": steps - start}


# Comfy's own LoraLoader accepts -100 to 100 and comfy.model_patcher.add_patches never clamps:
# the strength just scales the LoRA delta linearly into the weights, so nothing below this app
# objects to a large value. This bound exists to catch a typo (20 for 2.0), not to mark where the
# output stops being good. That point is per-LoRA and well under 2.0 for most of them, which is
# what user-authored prompting notes can describe.
LORA_STRENGTH_LIMIT = 100.0

# Metadata is user-authored and machine-local, keyed by the filename without its suffix.
# There are no built-in triggers or training claims, including for official adapters.
LORA_METADATA_LIMITS = {"trigger": 2000, "summary": 300, "prompting_notes": 12000}


def validate_lora_metadata(name: str, value: Any) -> dict[str, str]:
    if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError("LoRA metadata needs a filename without its directory or .safetensors suffix")
    if not isinstance(value, dict) or set(value) - LORA_METADATA_LIMITS.keys():
        raise ValueError("LoRA metadata accepts trigger, summary, and prompting_notes only")
    result = {}
    for field, limit in LORA_METADATA_LIMITS.items():
        text = value.get(field, "")
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError(f"LoRA {field} must be text no longer than {limit:,} characters")
        result[field] = text.strip()
    return result


def lora_metadata(settings: dict[str, Any], name: str) -> dict[str, str]:
    entries = settings.get("lora_metadata", {})
    if not isinstance(entries, dict):
        raise ValueError("lora_metadata must be an object keyed by LoRA filename without its suffix")
    return validate_lora_metadata(name, entries.get(name, {}))


def enhancer_lora_notes(settings: dict[str, Any], names: list[str]) -> list[str]:
    notes = []
    for name in dict.fromkeys(names):
        metadata = lora_metadata(settings, name)
        if metadata["summary"] or metadata["prompting_notes"]:
            notes.append(f"{name}: {metadata['summary']}\n{metadata['prompting_notes']}".strip())
    return notes


def lora_strength_error() -> str:
    return (
        f"LoRA strength must be a finite number from -{LORA_STRENGTH_LIMIT:g} to "
        f"{LORA_STRENGTH_LIMIT:g}. Around 1.0 is normal and most LoRAs degrade well before 2.0; "
        "anything higher is deliberate experimentation."
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_path() -> Path:
    return Path(os.environ.get("OFFCUT_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()


def load_settings() -> dict[str, Any]:
    path = config_path()
    user_settings = {}
    if path.exists():
        try:
            with SETTINGS_LOCK:
                user_settings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read settings at {path}: {exc}") from exc
    if not isinstance(user_settings, dict):
        raise RuntimeError(f"Settings at {path} must contain a JSON object")
    # Saved paths remain authoritative. An environment root supplies the default for a
    # fresh installation; missing model paths follow the selected root, not this checkout.
    comfy_root = Path(user_settings.get("comfy_root") or os.environ.get("OFFCUT_COMFY_ROOT") or DEFAULT_COMFY_ROOT).expanduser().resolve()
    defaults = deep_merge(DEFAULT_SETTINGS, {
        "comfy_root": str(comfy_root),
        "text_encoder": str(comfy_root / "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
        "vae": str(comfy_root / "models/vae/qwen_image_vae.safetensors"),
        "lora_dirs": [str(APP_ROOT / "models/loras"), str(comfy_root / "models/loras")],
    })
    settings = deep_merge(defaults, user_settings)
    settings["cover"]["preset"] = migrate_preset(settings["cover"]["preset"])
    return settings


def save_settings(settings: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(settings, indent=2, sort_keys=True) + "\n"
    with SETTINGS_LOCK:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    return path


def set_nested_value(settings: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError("Setting name must be a dot-separated key")
    target = settings
    for part in parts[:-1]:
        current = target.get(part)
        if current is None:
            current = {}
            target[part] = current
        if not isinstance(current, dict):
            raise ValueError(f"{part!r} is not a settings object")
        target = current
    target[parts[-1]] = value


def parse_setting_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(download: Download, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == download.size:
        if sha256_file(destination) == download.sha256:
            print(f"Already present: {destination}", file=sys.stderr)
            return
        raise RuntimeError(f"Existing file failed SHA-256 verification: {destination}")

    partial = destination.with_name(destination.name + ".part")
    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "5",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        download.url,
    ]
    print(f"Downloading {download.name} to {destination}", file=sys.stderr)
    subprocess.run(command, check=True)
    if partial.stat().st_size != download.size:
        raise RuntimeError(
            f"Downloaded size mismatch for {download.name}: "
            f"expected {download.size}, got {partial.stat().st_size}"
        )
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != download.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {download.name}: expected {download.sha256}, got {actual_sha256}"
        )
    partial.replace(destination)


def model_destination(settings: dict[str, Any], download: Download) -> Path:
    model_dir = Path(settings["model_dir"]).expanduser()
    folder = "loras" if download.relative_path.startswith("loras/") else "diffusion_models"
    return model_dir / folder / Path(download.relative_path).name


def resolve_checkpoint(settings: dict[str, Any], preset: Preset) -> Path:
    return Path(settings["model_dir"]).expanduser() / "diffusion_models" / preset.checkpoint


def split_lora_spec(spec: str) -> tuple[str, float]:
    name, separator, possible_strength = spec.rpartition(":")
    if separator:
        try:
            return name, float(possible_strength)
        except ValueError:
            pass
    return spec, 1.0


def resolve_lora(settings: dict[str, Any], spec: str) -> tuple[Path, float, str | None]:
    name, strength = split_lora_spec(spec)
    if not math.isfinite(strength) or not -LORA_STRENGTH_LIMIT <= strength <= LORA_STRENGTH_LIMIT:
        raise ValueError(lora_strength_error())
    candidate = Path(name).expanduser()
    candidates = [candidate]
    if candidate.suffix != ".safetensors":
        candidates.append(candidate.with_suffix(".safetensors"))
    for lora_dir in settings["lora_dirs"]:
        base = Path(lora_dir).expanduser()
        candidates.append(base / candidate.name)
        if candidate.suffix != ".safetensors":
            candidates.append(base / f"{candidate.name}.safetensors")
    for path in candidates:
        if path.is_file():
            return path.resolve(), strength, lora_metadata(settings, path.stem)["trigger"] or None
    raise FileNotFoundError(f"Could not find LoRA {name!r} in configured LoRA directories")


def prefix_prompt(prompt: str, triggers: list[str]) -> str:
    prompt = " ".join(prompt.split())
    unique_triggers = list(dict.fromkeys(trigger.strip() for trigger in triggers if trigger.strip()))
    if not unique_triggers:
        return prompt
    trigger_prefix = ", ".join(unique_triggers)
    if prompt.lower().startswith(trigger_prefix.lower()):
        return prompt
    return f"{trigger_prefix}, {prompt}"


def raw_sampling_shift(width: int, height: int) -> float:
    sequence_length = (width // 16) * (height // 16)
    low_tokens = (256 // 16) ** 2
    high_tokens = (1280 // 16) ** 2
    return 0.5 + (1.15 - 0.5) * (sequence_length - low_tokens) / (high_tokens - low_tokens)


def clean_enhanced_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    for prefix in ("Final prompt:", "Prompt:", "Enhanced prompt:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def is_opencode_go_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return parsed.hostname == "opencode.ai" and parsed.path.startswith("/zen/go/")


def enhance_prompt(prompt: str, trigger_prefix: str, settings: dict[str, Any], *, lora_names: list[str] | None = None) -> str:
    enhancer = settings["enhancer"]
    key_env = enhancer["api_key_env"]
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Prompt enhancement requires the {key_env} environment variable")

    context = ["Raw image prompt:", prompt]
    notes = enhancer_lora_notes(settings, lora_names or [])
    if notes:
        context.extend(["", "User-provided LoRA prompting suggestions (not requirements or verified training facts):", *notes])
    if trigger_prefix:
        context.extend(
            [
                "",
                "Active LoRA trigger/style phrase:",
                trigger_prefix,
                "",
                "The final prompt must keep this exact trigger phrase at the beginning and must not contradict its style.",
            ]
        )
    payload = {
        "model": enhancer["model"],
        "messages": [
            {"role": "system", "content": enhancer["system_prompt"]},
            {"role": "user", "content": "\n".join(context)},
        ],
        "temperature": float(enhancer["temperature"]),
        "max_tokens": int(enhancer["max_tokens"]),
    }
    request = urllib.request.Request(
        enhancer["endpoint"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "offcut-cli/1.0",
            # Enhancement is a standalone one-request conversation.
            **({"x-opencode-session": str(uuid.uuid4())} if is_opencode_go_endpoint(enhancer["endpoint"]) else {}),
        },
        method="POST",
    )
    class NoCredentialRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise RuntimeError(f"Prompt endpoint redirected to {newurl}; redirects are not allowed")

    try:
        opener = urllib.request.build_opener(NoCredentialRedirects)
        with opener.open(request, timeout=float(enhancer["timeout_seconds"])) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Prompt enhancement failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Prompt enhancement failed: {exc.reason}") from exc

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"OpenAI-compatible endpoint returned an invalid response: {response_data}") from exc
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    enhanced = clean_enhanced_prompt(str(content))
    if not enhanced:
        raise RuntimeError("OpenAI-compatible endpoint returned an empty prompt")
    if trigger_prefix and not enhanced.lower().startswith(trigger_prefix.lower()):
        enhanced = f"{trigger_prefix}, {enhanced}"
    return enhanced


# Comfy signals a stop by setting one global flag that its op wrapper checks on every forward
# and clears as it raises. The exception is a BaseException, deliberately, so nothing in the
# sampling stack swallows it; this converts it at the engine boundary into an ordinary error the
# server and the CLI can handle like any other failed run.
class GenerationCancelled(RuntimeError):
    pass


class Runtime:
    def __init__(self, comfy_root: Path):
        if not (comfy_root / "comfy" / "sd.py").is_file():
            raise FileNotFoundError(f"Comfy inference runtime not found at {comfy_root}")
        sys.path.insert(0, str(comfy_root))

        from comfy.cli_args import args as comfy_args

        # Comfy defaults to PyTorch's cudaMallocAsync backend (ComfyUI/cuda_malloc.py). Its pool
        # holds freed blocks that Comfy's get_free_memory counts as available, but that the
        # driver will not always hand back for the next allocation, so model_use_more_vram
        # over-commits and OOMs with gigabytes apparently free. The native caching allocator
        # reuses its cache reliably, which is the accounting Comfy is written against.
        comfy_args.disable_cuda_malloc = True

        import cuda_malloc  # noqa: F401
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU available to PyTorch. Use ComfyUI's Python environment with "
                "CUDA PyTorch for NVIDIA or ROCm PyTorch for AMD, and check your GPU driver."
            )

        # ROCm uses torch.cuda too. Its older Triton builds lack the INT8 ConvRot
        # kernels Krea needs; use comfy-kitchen's PyTorch fallback on AMD.
        is_rocm = bool(torch.version.hip)
        comfy_args.enable_triton_backend = not is_rocm
        comfy_args.disable_triton_backend = is_rocm
        if is_rocm:
            logging.warning("AMD/ROCm inference is experimental; using PyTorch quantization kernels.")

        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        import comfy.model_management
        import comfy.model_sampling
        import comfy.sample
        import comfy.samplers
        import comfy.sd
        import comfy.utils

        self.torch = torch
        self.Image = Image
        self.PngInfo = PngInfo
        self.mm = comfy.model_management
        self.model_sampling = comfy.model_sampling
        self.sample = comfy.sample
        self.samplers = comfy.samplers
        self.sd = comfy.sd
        self.utils = comfy.utils

    def synchronize(self) -> None:
        self.mm.synchronize()

    def request_interrupt(self) -> None:
        self.mm.interrupt_current_processing(True)

    def clear_interrupt(self) -> None:
        """Drop a stop flag nothing consumed.

        Comfy clears the flag only inside the check that raises, so a stop that lands after the
        last op of a run stays armed and would abort the next one at its first forward.
        """
        self.mm.interrupt_current_processing(False)


# Comfy hands conditioning back on intermediate_device, which is the CPU, so what a cache entry
# costs is host RAM rather than any of the VRAM the run is fighting over. A miss is what is
# expensive: the text encoder is paged onto the card for the encode, the diffusion checkpoint is
# evicted to make room for it, and the checkpoint is paged back for sampling. Holding the last few
# prompts makes returning to one free, which is the usual shape of iterating -- A, then B, then A
# again. The byte ceiling is the bound that actually matters here; prompts run to 20,000
# characters and one of those encodes to tens of megabytes on its own.
CONDITIONING_CACHE_ENTRIES = 8
CONDITIONING_CACHE_BYTES = 256 * 1024 * 1024


def conditioning_bytes(runtime: Runtime, conditioning: list[Any]) -> int:
    total = 0
    for tensor, metadata in conditioning:
        for value in (tensor, *metadata.values()):
            if runtime.torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def zero_conditioning(runtime: Runtime, conditioning: list[Any]) -> list[Any]:
    output = []
    for tensor, metadata in conditioning:
        metadata = metadata.copy()
        for key in ("pooled_output", "conditioning_lyrics"):
            value = metadata.get(key)
            if runtime.torch.is_tensor(value):
                metadata[key] = runtime.torch.zeros_like(value)
        output.append([runtime.torch.zeros_like(tensor), metadata])
    return output


def patch_count(model: Any) -> int:
    patches = getattr(model, "patches", {})
    return sum(len(value) if isinstance(value, list) else 1 for value in patches.values())


def gib(value: int) -> float:
    return round(value / (1024**3), 3)


def file_fingerprint(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


class KreaEngine:
    def __init__(
        self,
        runtime: Runtime,
        settings: dict[str, Any],
        preset: Preset,
        loras: list[tuple[Path, float]],
        width: int,
        height: int,
    ):
        self.runtime = runtime
        self.settings = settings
        self.preset = preset
        self.width = width
        self.height = height
        self.load_started = time.perf_counter()

        torch = runtime.torch
        torch.cuda.reset_peak_memory_stats()
        checkpoint = resolve_checkpoint(settings, preset)
        text_encoder = Path(settings["text_encoder"]).expanduser()
        vae_path = Path(settings["vae"]).expanduser()
        for path, label in (
            (checkpoint, "diffusion checkpoint"),
            (text_encoder, "Krea text encoder"),
            (vae_path, "VAE"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

        # The base patcher is never patched, so a LoRA change can re-clone from it instead of
        # re-reading the checkpoint. Comfy LoRAs are ModelPatcher patches, not baked weights.
        self.base_model = runtime.sd.load_diffusion_model(str(checkpoint))
        self.model = self.base_model
        self.applied_loras: list[dict[str, Any]] = []
        # At most the two stacks for a recipe. They share base_model's module and preserve
        # patches_uuid across route changes; only Comfy's resident weight repatch remains.
        self.lora_stacks: OrderedDict[tuple[Any, ...], tuple[Any, list[dict[str, Any]]]] = OrderedDict()

        self.clip = runtime.sd.load_clip(
            ckpt_paths=[str(text_encoder)],
            embedding_directory=[],
            clip_type=runtime.sd.CLIPType.KREA2,
        )
        vae_sd, vae_metadata = runtime.utils.load_torch_file(str(vae_path), return_metadata=True)
        self.vae = runtime.sd.VAE(sd=vae_sd, metadata=vae_metadata)
        self.vae.throw_exception_if_invalid()
        self.conditioning_cache: OrderedDict[
            tuple[str, str, bool], tuple[list[Any], list[Any]]
        ] = OrderedDict()
        self.set_loras(loras)
        runtime.synchronize()
        self.load_seconds = time.perf_counter() - self.load_started
        self.load_peak_allocated = gib(torch.cuda.max_memory_allocated())
        self.load_peak_reserved = gib(torch.cuda.max_memory_reserved())

    def _sampling_model(self, model: Any, shift: float) -> Any:
        runtime = self.runtime

        class KreaSampling(runtime.model_sampling.ModelSamplingFlux, runtime.model_sampling.CONST):
            pass

        sampling = KreaSampling(model.model.model_config)
        sampling.set_parameters(shift=shift)
        patched = model.clone()
        patched.add_object_patch("model_sampling", sampling)
        return patched

    def set_preset(self, preset: Preset) -> None:
        if preset.checkpoint != self.preset.checkpoint:
            raise ValueError("Changing checkpoints requires a new engine")
        self.preset = preset

    def set_loras(self, loras: list[tuple[Path, float]], on_change: Any | None = None) -> None:
        """Swap the LoRA stack without reloading anything from disk.

        Comfy applies LoRAs as ModelPatcher patches on a clone that shares the underlying
        module, and ModelPatcher.partially_load re-patches resident weights when patches_uuid
        changes. So a new stack only costs a clone plus a repatch of the weights already on the
        GPU. CLIP is never patched (strength_clip is 0), which is also why the conditioning
        cache stays valid across a swap.
        """
        requested = list(loras)
        turbo = []
        if self.preset.turbo_lora:
            turbo_lora = Path(self.settings["model_dir"]).expanduser() / "loras" / DOWNLOADS["turbo-lora"].relative_path.split("/")[-1]
            if not turbo_lora.is_file():
                raise FileNotFoundError(f"Missing Turbo LoRA: {turbo_lora}")
            turbo = [(turbo_lora, 1.0)]

        # Keep the Turbo adapter first, matching the old Turbo route's patch order. Both
        # hybrid stacks start from the never-patched base, so styles enter each exactly once.
        raw_model = None
        if not self.preset.turbo:
            raw_model, raw_details = self._lora_stack(requested, on_change)
        if self.preset.turbo_lora:
            turbo_model, turbo_details = self._lora_stack([*turbo, *requested], on_change)
            self.turbo_model = self._sampling_model(turbo_model, 1.15)
        if raw_model is not None:
            self.raw_model = self._sampling_model(raw_model, raw_sampling_shift(self.width, self.height))
        self.model = self.turbo_model if self.preset.turbo else self.raw_model
        self.applied_loras = turbo_details if self.preset.turbo else raw_details

    def _lora_stack(self, requested: list[tuple[Path, float]], on_change: Any | None) -> tuple[Any, list[dict[str, Any]]]:
        runtime = self.runtime

        for lora_path, _ in requested:
            if not lora_path.is_file():
                raise FileNotFoundError(f"Missing LoRA: {lora_path}")

        # Fingerprints, not just paths, so editing a LoRA file in place still re-reads it.
        lora_key = tuple((strength, file_fingerprint(path)) for path, strength in requested)
        if lora_key in self.lora_stacks:
            self.lora_stacks.move_to_end(lora_key)
            return self.lora_stacks[lora_key]
        if on_change is not None:
            on_change()

        # Built into locals first: a LoRA that fails to load leaves the engine on its old stack.
        model = self.base_model
        applied_loras = []
        for lora_path, strength in requested:
            before = patch_count(model)
            lora_sd, lora_metadata = runtime.utils.load_torch_file(
                str(lora_path), safe_load=True, return_metadata=True
            )
            model, _ = runtime.sd.load_lora_for_models(
                model=model,
                clip=None,
                lora=lora_sd,
                strength_model=strength,
                strength_clip=0.0,
                lora_metadata=lora_metadata,
            )
            after = patch_count(model)
            if after <= before:
                raise RuntimeError(f"LoRA did not patch any Krea model weights: {lora_path}")
            applied_loras.append({"path": str(lora_path), "strength": strength, "patches": after - before})

        self.lora_stacks[lora_key] = (model, applied_loras)
        while len(self.lora_stacks) > 2:
            self.lora_stacks.popitem(last=False)
        return model, applied_loras

    def set_dimensions(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width = width
        self.height = height
        if not self.preset.turbo and hasattr(self, "raw_model"):
            self.raw_model = self._sampling_model(self.raw_model, raw_sampling_shift(width, height))
            self.model = self.raw_model

    def generate(
        self,
        prompt: str,
        raw_prompt: str,
        output_dir: Path,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str,
        enhanced_prompt: str = "",
        progress_callback: Any | None = None,
        extra_metadata: dict[str, Any] | None = None,
        raw_portion: float | None = None,
        raw_steps: int | None = None,
    ) -> dict[str, Any]:
        torch = self.runtime.torch
        try:
            with torch.no_grad():
                try:
                    return self._generate(
                        prompt,
                        raw_prompt,
                        output_dir,
                        seed,
                        steps,
                        guidance,
                        negative_prompt,
                        enhanced_prompt,
                        progress_callback,
                        extra_metadata,
                        raw_portion,
                        raw_steps,
                    )
                except torch.OutOfMemoryError:
                    pass  # Fall through to one retry, below.
                # Only reachable after an OOM. The retry happens outside the except block
                # because the live traceback holds the failed run's frame, and with it every
                # tensor that run allocated; trimming while it is alive frees nothing.
                self.runtime.mm.soft_empty_cache()
                return self._generate(
                    prompt,
                    raw_prompt,
                    output_dir,
                    seed,
                    steps,
                    guidance,
                    negative_prompt,
                    enhanced_prompt,
                    progress_callback,
                    extra_metadata,
                    raw_portion,
                    raw_steps,
                )
        except self.runtime.mm.InterruptProcessingException:
            # Comfy's op wrapper raises this from inside whichever forward was running when the
            # stop arrived, so it can surface out of encoding, sampling, or decoding alike.
            raise GenerationCancelled("Generation stopped") from None
        finally:
            # PyTorch caches freed blocks rather than returning them to the driver, so without
            # this trim a long-lived server holds its highest-resolution run's high-water mark
            # forever and starves everything else on the card.
            self.runtime.mm.soft_empty_cache()

    def _remember_conditioning(
        self,
        cache_key: tuple[str, str, bool],
        positive: list[Any],
        negative: list[Any],
    ) -> None:
        """Keep this encode and evict the least recently used entries over either ceiling.

        The single surviving entry is never evicted even when it alone is over the byte ceiling:
        the run about to sample is holding it anyway, so dropping it would buy no memory back and
        would only guarantee a re-encode if that same prompt came round again.
        """
        self.conditioning_cache[cache_key] = (positive, negative)
        self.conditioning_cache.move_to_end(cache_key)
        cached_bytes = sum(
            conditioning_bytes(self.runtime, cached_positive)
            + conditioning_bytes(self.runtime, cached_negative)
            for cached_positive, cached_negative in self.conditioning_cache.values()
        )
        while len(self.conditioning_cache) > 1 and (
            len(self.conditioning_cache) > CONDITIONING_CACHE_ENTRIES
            or cached_bytes > CONDITIONING_CACHE_BYTES
        ):
            _, (dropped_positive, dropped_negative) = self.conditioning_cache.popitem(last=False)
            cached_bytes -= conditioning_bytes(self.runtime, dropped_positive)
            cached_bytes -= conditioning_bytes(self.runtime, dropped_negative)

    def _sample_two_stage(
        self, latent: Any, noise: Any, positive: Any, negative: Any, seed: int,
        steps: int, guidance: float, recipe: dict[str, Any], progress: Any,
    ) -> tuple[Any, dict[str, Any]]:
        runtime = self.runtime

        def schedule(model: Any, density: int) -> Any:
            return runtime.samplers.KSampler(
                model, steps=density, device=model.load_device, sampler="euler",
                scheduler="simple", denoise=1.0, model_options=model.model_options,
            ).sigmas.detach().cpu()

        raw_sigmas, turbo_sigmas = split_sigma_schedules(
            schedule(self.raw_model, recipe["raw_steps"]), schedule(self.turbo_model, steps), recipe["raw_portion"],
        )
        raw_count, turbo_count = len(raw_sigmas) - 1, len(turbo_sigmas) - 1
        raw_cost = 2 if guidance else 1
        total_cost = raw_count * raw_cost + turbo_count

        def run(model: Any, sigmas: Any, frame: Any, stage_noise: Any, cfg: float, conditioning: Any, stage: str, offset: int, cost: int, disable_noise: bool) -> Any:
            runtime.mm.throw_exception_if_processing_interrupted()

            def callback(step: int, _x0: Any, _x: Any, total: int) -> None:
                progress("sampling", 0.30 + 0.56 * (offset + (step + 1) * cost) / total_cost,
                         f"Raw {step + 1 if stage == 'Raw' else raw_count}/{raw_count} → Turbo {step + 1 if stage == 'Turbo' else 0}/{turbo_count}")

            progress("sampling", 0.30 + 0.56 * offset / total_cost,
                     f"Raw {0 if stage == 'Raw' else raw_count}/{raw_count} → Turbo 0/{turbo_count} · Preparing {stage}")
            return runtime.sample.sample(
                model=model, noise=stage_noise, steps=len(sigmas) - 1, cfg=cfg,
                sampler_name="euler", scheduler="simple", positive=positive, negative=conditioning,
                latent_image=frame, denoise=1.0, disable_noise=disable_noise, force_full_denoise=False,
                sigmas=sigmas.to(model.load_device), disable_pbar=True, seed=seed, callback=callback,
            )

        intermediate = run(self.raw_model, raw_sigmas, latent, noise, guidance + 1.0, negative, "Raw", 0, raw_cost, False)
        samples = run(self.turbo_model, turbo_sigmas, intermediate, runtime.torch.zeros_like(intermediate),
                      1.0, zero_conditioning(runtime, positive), "Turbo", raw_count * raw_cost, 1, True)
        return samples, {
            "raw_executed_steps": raw_count, "turbo_executed_steps": turbo_count,
            "handoff_sigma": float(raw_sigmas[-1]), "raw_cfg": guidance + 1.0, "turbo_cfg": 1.0,
            "raw_shift": raw_sampling_shift(self.width, self.height), "turbo_shift": 1.15,
            "sampler": "euler", "scheduler": "simple",
        }

    def _generate(
        self,
        prompt: str,
        raw_prompt: str,
        output_dir: Path,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str,
        enhanced_prompt: str,
        progress_callback: Any | None,
        extra_metadata: dict[str, Any] | None,
        raw_portion: float | None,
        raw_steps: int | None,
    ) -> dict[str, Any]:
        runtime = self.runtime
        torch = runtime.torch
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        generation_started = time.perf_counter()
        recipe = hybrid_settings(self.preset, steps, raw_portion, raw_steps)
        if self.preset.turbo:
            if guidance != 0.0:
                raise ValueError("Turbo guidance is fixed at 0.0")
            negative_prompt = ""

        def progress(stage: str, fraction: float, detail: str) -> None:
            if progress_callback is not None:
                progress_callback(stage, fraction, detail)

        cache_key = (prompt, negative_prompt, self.preset.turbo)
        progress("encoding", 0.18, "Encoding Krea conditioning")
        phase_started = time.perf_counter()
        if cache_key in self.conditioning_cache:
            positive, negative = self.conditioning_cache[cache_key]
            self.conditioning_cache.move_to_end(cache_key)
            conditioning_cache_hit = True
        else:
            positive = self.clip.encode_from_tokens_scheduled(
                self.clip.tokenize(prompt), show_pbar=False
            )
            if self.preset.turbo:
                negative = zero_conditioning(runtime, positive)
            else:
                negative = self.clip.encode_from_tokens_scheduled(
                    self.clip.tokenize(negative_prompt), show_pbar=False
                )
            self._remember_conditioning(cache_key, positive, negative)
            runtime.mm.unload_model_and_clones(self.clip.patcher)
            runtime.synchronize()
            conditioning_cache_hit = False
        encode_seconds = time.perf_counter() - phase_started

        latent = torch.zeros(
            [1, 4, self.height // 8, self.width // 8],
            device=runtime.mm.intermediate_device(),
            dtype=runtime.mm.intermediate_dtype(),
        )
        latent = runtime.sample.fix_empty_latent_channels(
            self.model, latent, downscale_ratio_spacial=8
        )
        noise = runtime.sample.prepare_noise(latent, seed)

        phase_started = time.perf_counter()
        progress("sampling", 0.30, f"Sampling step 0 of {steps}")

        def sampler_progress(step: int, _x0: Any, _x: Any, total_steps: int) -> None:
            completed = min(step + 1, total_steps)
            fraction = 0.30 + 0.56 * completed / max(total_steps, 1)
            progress("sampling", fraction, f"Sampling step {completed} of {total_steps}")

        sampling_details = {}
        if self.preset.two_stage:
            samples, sampling_details = self._sample_two_stage(
                latent, noise, positive, negative, seed, steps, guidance, recipe, progress,
            )
        else:
            samples = runtime.sample.sample(
                model=self.model,
                noise=noise,
                steps=steps,
                cfg=guidance + 1.0,
                sampler_name="euler",
                scheduler="simple",
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1.0,
                disable_pbar=True,
                seed=seed,
                callback=sampler_progress,
            )
        runtime.synchronize()
        sample_seconds = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        progress("decoding", 0.90, "Decoding pixels")
        # Comfy's tiled fallback returns inference tensors and process_output mutates them.
        # Keep that whole decode inside the context, including the OOM fallback's postprocess.
        with torch.inference_mode():
            images = self.vae.decode(samples)
        runtime.synchronize()
        decode_seconds = time.perf_counter() - phase_started
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_path = (output_dir / f"{timestamp}_{self.preset.name}_seed-{seed}.png").resolve()
        progress("saving", 0.97, "Writing PNG and metadata")
        image_array = (images[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        png_info = runtime.PngInfo()
        metadata = {
            "preset": self.preset.name,
            "seed": seed,
            "width": self.width,
            "height": self.height,
            "steps": steps,
            "guidance": guidance,
            "negative_prompt": negative_prompt,
            **recipe,
            **({"sampling": sampling_details} if sampling_details else {}),
            "raw_prompt": raw_prompt,
            "enhanced_prompt": enhanced_prompt,
            "final_prompt": prompt,
            "loras": self.applied_loras,
            "conditioning_cache_hit": conditioning_cache_hit,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        png_info.add_text("krea2", json.dumps(metadata, separators=(",", ":")))
        runtime.Image.fromarray(image_array).save(output_path, pnginfo=png_info)
        progress("publishing", 0.99, "Publishing image to its board")

        total_seconds = time.perf_counter() - generation_started
        return {
            **metadata,
            "output_path": str(output_path),
            "timings_seconds": {
                "load": round(self.load_seconds, 3),
                "encode": round(encode_seconds, 3),
                "sample": round(sample_seconds, 3),
                "decode": round(decode_seconds, 3),
                "generation_total": round(total_seconds, 3),
            },
            "vram_gib": {
                "load_peak_allocated": self.load_peak_allocated,
                "load_peak_reserved": self.load_peak_reserved,
                "generation_peak_allocated": gib(torch.cuda.max_memory_allocated()),
                "generation_peak_reserved": gib(torch.cuda.max_memory_reserved()),
            },
        }

    def close(self) -> None:
        self.runtime.mm.unload_all_models()
        self.runtime.mm.soft_empty_cache()


def validate_dimensions(width: int, height: int) -> None:
    if width < 256 or height < 256:
        raise ValueError("Width and height must be at least 256 pixels")
    if width % 16 or height % 16:
        raise ValueError("Width and height must be divisible by 16")
    if width > 2048 or height > 2048 or width * height > 4_194_304:
        raise ValueError("Width and height are limited to 2048 pixels and 4,194,304 total pixels")


def validate_generation_values(prompt: str, seed: int, steps: int, guidance: float) -> None:
    if not prompt.strip() or len(prompt) > 20_000:
        raise ValueError("Prompt must contain between 1 and 20,000 characters")
    if not 0 <= seed < 2**63:
        raise ValueError("Seed must be between 0 and 2^63 - 1")
    if not 1 <= steps <= 100:
        raise ValueError("Steps must be between 1 and 100")
    if not math.isfinite(guidance) or not 0.0 <= guidance <= 20.0:
        raise ValueError("Guidance must be a finite number from 0.0 to 20.0")


def prepare_generation(args: argparse.Namespace, settings: dict[str, Any]) -> tuple[Preset, list[tuple[Path, float]], str, str]:
    preset = PRESETS[migrate_preset(args.preset)]
    resolved_loras = [resolve_lora(settings, spec) for spec in args.lora]
    turbo_lora_name = Path(DOWNLOADS["turbo-lora"].relative_path).stem
    if any(path.stem == turbo_lora_name for path, _, _ in resolved_loras):
        raise ValueError(
            "The Turbo LoRA is managed by the Turbo and Raw → Turbo presets and cannot be passed through --lora"
        )
    loras = [(path, strength) for path, strength, _ in resolved_loras]
    triggers = [trigger for _, _, trigger in resolved_loras if trigger]
    triggers.extend(args.trigger)
    trigger_prefix = ", ".join(dict.fromkeys(triggers))
    prompt = prefix_prompt(args.prompt, triggers)
    enhance = settings["enhancer"]["enabled"] if args.enhance is None else args.enhance
    if enhance:
        prompt = enhance_prompt(prompt, trigger_prefix, settings, lora_names=[path.stem for path, _, _ in resolved_loras])
    return preset, loras, prompt, trigger_prefix


def run_generate(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    validate_dimensions(args.width, args.height)
    preset, loras, prompt, _ = prepare_generation(args, settings)
    steps = preset.default_steps if args.steps is None else args.steps
    guidance = preset.default_guidance if args.guidance is None else args.guidance
    validate_generation_values(args.prompt, args.seed, steps, guidance)
    recipe = hybrid_settings(preset, steps, args.raw_portion, args.raw_steps)
    if preset.turbo and guidance != 0.0:
        raise ValueError("Turbo guidance is fixed at 0.0")
    runtime = Runtime(Path(settings["comfy_root"]).expanduser())
    engine = KreaEngine(runtime, settings, preset, loras, args.width, args.height)
    try:
        result = engine.generate(
            prompt=prompt,
            raw_prompt=args.prompt,
            output_dir=Path(args.output_dir or settings["output_dir"]).expanduser(),
            seed=args.seed,
            steps=steps,
            guidance=guidance,
            negative_prompt=args.negative_prompt,
            **recipe,
        )
    finally:
        engine.close()
    print(json.dumps(result, indent=2, sort_keys=True))


def run_benchmark(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    validate_dimensions(args.width, args.height)
    runtime = Runtime(Path(settings["comfy_root"]).expanduser())
    output_dir = Path(args.output_dir or settings["output_dir"]).expanduser() / "benchmarks"
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "runs_per_preset": args.runs,
        "presets": {},
    }
    for preset_name in args.presets:
        preset = PRESETS[preset_name]
        engine = KreaEngine(runtime, settings, preset, [], args.width, args.height)
        try:
            runs = []
            for run_index in range(args.runs):
                result = engine.generate(
                    prompt=args.prompt,
                    raw_prompt=args.prompt,
                    output_dir=output_dir,
                    seed=args.seed,
                    steps=preset.default_steps,
                    guidance=preset.default_guidance,
                    negative_prompt="",
                )
                result["run"] = run_index + 1
                runs.append(result)
                print(
                    f"{preset_name} run {run_index + 1}/{args.runs}: {result['output_path']}",
                    file=sys.stderr,
                )
            report["presets"][preset_name] = runs
        finally:
            engine.close()
            del engine

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").resolve()
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_download(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    if "all" in args.names and len(args.names) > 1:
        raise ValueError("The download target 'all' cannot be combined with individual targets")
    names = list(DOWNLOADS) if not args.names or args.names == ["all"] else args.names
    paths = []
    for name in names:
        download = DOWNLOADS[name]
        destination = model_destination(settings, download)
        download_file(download, destination)
        paths.append(str(destination.resolve()))
    print(json.dumps({"downloaded": paths}, indent=2))


def run_models(settings: dict[str, Any]) -> None:
    models = {}
    for name, download in DOWNLOADS.items():
        path = model_destination(settings, download)
        models[name] = {
            "path": str(path),
            "present": path.is_file(),
            "size_ok": path.is_file() and path.stat().st_size == download.size,
        }
    for name, path_value in (("text-encoder", settings["text_encoder"]), ("vae", settings["vae"])):
        path = Path(path_value).expanduser()
        models[name] = {"path": str(path), "present": path.is_file()}
    print(json.dumps(models, indent=2, sort_keys=True))


def run_settings(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    if args.settings_command == "show":
        print(json.dumps({"path": str(config_path()), "settings": settings}, indent=2, sort_keys=True))
        return
    if args.settings_command == "set":
        old_root = Path(settings["comfy_root"]).expanduser().resolve()
        set_nested_value(settings, args.key, parse_setting_value(args.value))
        if args.key == "comfy_root":
            new_root = Path(settings["comfy_root"]).expanduser().resolve()
            settings["comfy_root"] = str(new_root)
            # Move only conventional paths; custom model locations stay independent.
            for key, relative in (("text_encoder", "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"), ("vae", "models/vae/qwen_image_vae.safetensors")):
                if Path(settings[key]).expanduser().resolve() == old_root / relative:
                    settings[key] = str(new_root / relative)
            settings["lora_dirs"] = [str(new_root / "models/loras") if Path(value).expanduser().resolve() == old_root / "models/loras" else value for value in settings["lora_dirs"]]
        if args.key == "lora_metadata" or args.key.startswith("lora_metadata."):
            entries = settings["lora_metadata"]
            if not isinstance(entries, dict):
                raise ValueError("lora_metadata must be an object")
            settings["lora_metadata"] = {name: validate_lora_metadata(name, value) for name, value in entries.items()}
        path = save_settings(settings)
        print(json.dumps({"path": str(path), "updated": args.key}, indent=2))
        return
    raise RuntimeError("Missing settings command")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offcut-cli", description="Offcut CLI: generate images with Krea 2 using ComfyUI's runtime")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose runtime logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one image and print its absolute path as JSON")
    generate.add_argument("prompt")
    generate.add_argument("--preset", type=migrate_preset, choices=PRESETS, default=DEFAULT_PRESET)
    generate.add_argument("--width", type=int, default=1024)
    generate.add_argument("--height", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--steps", type=int)
    generate.add_argument("--guidance", type=float, help="Krea-native guidance; Turbo defaults to 0.0")
    generate.add_argument("--negative-prompt", default="", help="Used by Raw and the raw stage of Raw → Turbo")
    generate.add_argument("--raw-portion", type=float, help="Raw → Turbo handoff percentage (default 8); not a share of runtime")
    generate.add_argument("--raw-steps", type=int, help="Raw → Turbo raw schedule density (default 52); --steps sets turbo density (default 12)")
    generate.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    generate.add_argument("--trigger", action="append", default=[], help="Additional exact style trigger to prefix")
    generate.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None)
    generate.add_argument("--output-dir")

    benchmark = subparsers.add_parser("benchmark", help="Run matched cold/warm generations and save a JSON report")
    benchmark.add_argument("--prompt", default="A red fox sitting in fresh snow, golden hour, photorealistic")
    benchmark.add_argument(
        "--presets",
        nargs="+",
        choices=PRESETS,
        default=[DEFAULT_PRESET, HYBRID_PRESET],
    )
    benchmark.add_argument("--runs", type=int, default=2)
    benchmark.add_argument("--width", type=int, default=1024)
    benchmark.add_argument("--height", type=int, default=1024)
    benchmark.add_argument("--seed", type=int, default=12345)
    benchmark.add_argument("--output-dir")

    download = subparsers.add_parser("download", help="Resume and verify the required model downloads")
    download.add_argument("names", nargs="*", choices=[*DOWNLOADS, "all"])

    subparsers.add_parser("models", help="Show model paths and availability")

    settings_parser = subparsers.add_parser("settings", help="View or update persistent non-secret settings")
    settings_subparsers = settings_parser.add_subparsers(dest="settings_command", required=True)
    settings_subparsers.add_parser("show")
    settings_set = settings_subparsers.add_parser("set")
    settings_set.add_argument("key", help="Dot-separated key, for example enhancer.endpoint")
    settings_set.add_argument("value", help="A JSON value or plain string")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        settings = load_settings()
        if args.command == "generate":
            run_generate(args, settings)
        elif args.command == "benchmark":
            if args.runs < 1:
                raise ValueError("Benchmark runs must be at least 1")
            run_benchmark(args, settings)
        elif args.command == "download":
            run_download(args, settings)
        elif args.command == "models":
            run_models(settings)
        elif args.command == "settings":
            run_settings(args, settings)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
