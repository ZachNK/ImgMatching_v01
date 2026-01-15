# project/imatch/loading.py
"""Utility helpers for dataset-aware path handling and result exports."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

from PIL import Image
import torch
from torchvision import transforms

REPO_DIR = Path(os.getenv("REPO_DIR", "/workspace/dinov3"))
IMG_ROOT = Path(os.getenv("IMG_ROOT", "/opt/datasets"))
EMBED_ROOT = Path(os.getenv("EMBED_ROOT", "/exports/dinov3_embeds"))
MATCH_ROOT = Path(os.getenv("MATCH_ROOT", "/exports/dinov3_match"))
VIS_ROOT = Path(os.getenv("VIS_ROOT", "/exports/dinov3_vis"))
REFERENCE_ROOT = Path(os.getenv("REFERENCE_ROOT") or os.getenv("QUERY_ROOT", "/opt/references"))
REFERENCE_PREFIX = (os.getenv("REFERENCE_PREFIX") or os.getenv("QUERY_PREFIX") or "").strip()
REFERENCE_DATASET_PREFIX = (
    os.getenv("REFERENCE_DATASET_PREFIX")
    or os.getenv("QUERY_DATASET_PREFIX")
    or ""
    ).strip()
REFERENCE_EMBED_ROOT = Path(
    os.getenv("REFERENCE_EMBED_ROOT") or os.getenv("QUERY_EMBED_ROOT", "/exports/dinov3_reference_embeds")
)
# Backward-compatible aliases (use REFERENCE_* going forward).
QUERY_ROOT = REFERENCE_ROOT
QUERY_PREFIX = REFERENCE_PREFIX
QUERY_DATASET_PREFIX = REFERENCE_DATASET_PREFIX
QUERY_EMBED_ROOT = REFERENCE_EMBED_ROOT

DATASET_ROOT = IMG_ROOT
EXPORT_ROOT = Path("/exports")
WEIGHT_ROOT = Path("/workspace/weights")
JSON = Path("/workspace/project/json/data_key.json")

with JSON.open("r", encoding="utf-8") as s:
    registry = json.load(s)

DATASETS: Dict[str, Dict[str, Any]] = registry.get("datasets", {})
WEIGHT_SETS: Dict[str, Dict[str, Any]] = registry.get("weights", {})


def _first_key(data: Dict[str, Any]) -> str:
    return next(iter(data)) if data else ""


def _dataset_token(dataset_key: str | None) -> str:
    key = dataset_key or DATASET_KEY
    key_text = str(key).strip()
    if not key_text:
        raise ValueError("[loading] dataset_key is required to build dataset-aware export roots.")
    return _sanitize_token(key_text)


def dataset_embed_root(dataset_key: str | None = None, root: Path | None = None) -> Path:
    """
    Resolve the embedding export root for a dataset, adding a dataset subfolder
    under EMBED_ROOT unless the root is already dataset-specific.
    """
    base_root = Path(root) if root is not None else EMBED_ROOT
    token = _dataset_token(dataset_key)
    if _sanitize_token(base_root.name) == token:
        return base_root
    return base_root / token


def dataset_reference_embed_root(dataset_key: str | None = None, root: Path | None = None) -> Path:
    """
    Resolve the reference embedding export root for a dataset, adding a dataset
    subfolder under REFERENCE_EMBED_ROOT unless the root already ends with it.
    """
    base_root = Path(root) if root is not None else REFERENCE_EMBED_ROOT
    token = _dataset_token(dataset_key)
    if _sanitize_token(base_root.name) == token:
        return base_root
    return base_root / token

# Backward compatibility for legacy imports.
dataset_query_embed_root = dataset_reference_embed_root


def _normalize_label(value: Any) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return str(value)
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("[loading] Label value cannot be empty.")
    return text


def _sanitize_token(label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    return token or "group"


def _format_index(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:04d}"


IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")
_TRAILING_FRAME_RX = re.compile(r"(\d+)(?=\.[^.]+$)")
_FRAME_FILENAME_CACHE: Dict[Tuple[str, str, int], str] = {}


@dataclass(frozen=True)
class DatasetImageEntry:
    dataset_key: str
    capture_id: str
    label: str
    label_token: str
    folder: Path
    filename_template: str

    def folder_str(self) -> str:
        parts = [p for p in self.folder.parts if p and p != "."]
        return "/".join(parts)

    def build_filename(self, index: int) -> str:
        return _resolve_filename(self, index)


@dataclass(frozen=True)
class DatasetState:
    key: str
    entries_by_label: Dict[str, DatasetImageEntry]


@dataclass(frozen=True)
class ImageInstance:
    entry: DatasetImageEntry
    filename: str

    @property
    def dataset_key(self) -> str:
        return self.entry.dataset_key

    @property
    def capture_id(self) -> str:
        return self.entry.capture_id

    @property
    def label(self) -> str:
        return self.entry.label

    @property
    def label_token(self) -> str:
        return self.entry.label_token

    @property
    def folder(self) -> str:
        return self.entry.folder_str()

    @property
    def stem(self) -> str:
        return Path(self.filename).stem

    @property
    def path(self) -> Path:
        return DATASET_ROOT / self.entry.folder / self.filename


def _build_image_entry(
    dataset_key: str,
    dataset_cfg: Dict[str, Any],
    capture_id: str,
    raw_value: Any,
) -> DatasetImageEntry:
    base_folder_template = dataset_cfg.get("folder_template", "{capture_id}_{label}")
    base_file_template = dataset_cfg.get("filename_template", "{capture_id}_{label}_{frame}.jpg")
    root_prefix = dataset_cfg.get("root")
    if root_prefix in (None, ""):
        root_prefix = dataset_key

    override: Dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
    label_value = override.get("label")
    if label_value is None:
        label_value = override.get("altitude", raw_value)
    label = _normalize_label(label_value)
    label_token = _sanitize_token(label)

    folder_override = override.get("folder")
    folder_template = override.get("folder_template", base_folder_template)
    template_ctx = {
        "capture_id": capture_id,
        "label": label,
        "label_token": label_token,
        "dataset_key": dataset_key,
    }
    folder_rel = folder_override or folder_template.format(**template_ctx)
    folder_path = Path(root_prefix).joinpath(Path(folder_rel)) if root_prefix else Path(folder_rel)

    filename_template = override.get("filename_template", base_file_template)

    return DatasetImageEntry(
        dataset_key=dataset_key,
        capture_id=capture_id,
        label=label,
        label_token=label_token,
        folder=folder_path,
        filename_template=filename_template,
    )


def _build_frame_regex(template: str, entry: DatasetImageEntry) -> Optional[re.Pattern]:
    token_keys = ("frame", "index", "idx")
    if not any(f"{{{key}}}" in template for key in token_keys):
        return None
    pattern = re.escape(template)
    replacements = {
        "{capture_id}": entry.capture_id,
        "{label}": entry.label,
        "{label_token}": entry.label_token,
        "{dataset_key}": entry.dataset_key,
    }
    for placeholder, value in replacements.items():
        pattern = pattern.replace(re.escape(placeholder), re.escape(str(value)))
    for key in token_keys:
        pattern = pattern.replace(re.escape(f"{{{key}}}"), r"(?P<frame>\d+)")
    return re.compile(f"^{pattern}$", re.IGNORECASE)


def _find_existing_filename(entry: DatasetImageEntry, index: int) -> Optional[str]:
    folder_path = DATASET_ROOT / entry.folder
    if not folder_path.exists():
        return None

    idx_int = int(index)
    regex = _build_frame_regex(entry.filename_template, entry)

    def _matches(name: str) -> Optional[str]:
        if regex:
            m = regex.match(name)
            if m and "frame" in m.groupdict():
                try:
                    if int(m.group("frame")) == idx_int:
                        return m.group("frame")
                except ValueError:
                    pass
        m2 = _TRAILING_FRAME_RX.search(name)
        if m2:
            try:
                if int(m2.group(1)) == idx_int:
                    return m2.group(1)
            except ValueError:
                return None
        return None

    candidates: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(sorted(folder_path.glob(f"*.{ext}")))
    if not candidates:
        candidates = [p for p in folder_path.iterdir() if p.is_file()]

    for path in candidates:
        token = _matches(path.name)
        if token is not None:
            return path.name
    return None


def _extract_frame_token(name: str, fallback_index: int) -> str:
    m = _TRAILING_FRAME_RX.search(name)
    if m:
        return m.group(1)
    return _format_index(fallback_index)


def _resolve_filename(entry: DatasetImageEntry, index: int) -> str:
    cache_key = (entry.dataset_key, entry.label, int(index))
    cached = _FRAME_FILENAME_CACHE.get(cache_key)
    if cached:
        return cached

    existing = _find_existing_filename(entry, index)
    if existing:
        _FRAME_FILENAME_CACHE[cache_key] = existing
        return existing

    frame = _format_index(index)
    filename = entry.filename_template.format(
        capture_id=entry.capture_id,
        label=entry.label,
        label_token=entry.label_token,
        dataset_key=entry.dataset_key,
        index=int(index),
        idx=int(index),
        frame=frame,
    )
    _FRAME_FILENAME_CACHE[cache_key] = filename
    return filename


def _load_dataset_state(dataset_key: str) -> DatasetState:
    if dataset_key not in DATASETS:
        raise KeyError(f"[loading] Unknown dataset key: {dataset_key}")
    dataset_cfg = DATASETS[dataset_key]
    images_cfg = dataset_cfg.get("images") or dataset_cfg.get("captures")
    if not isinstance(images_cfg, dict) or not images_cfg:
        raise ValueError(f"[loading] Dataset '{dataset_key}' must define 'images'.")

    entries: Dict[str, DatasetImageEntry] = {}
    for capture_id, raw_value in images_cfg.items():
        entry = _build_image_entry(dataset_key, dataset_cfg, str(capture_id), raw_value)
        if entry.label in entries:
            raise ValueError(
                f"[loading] Duplicate label '{entry.label}' detected in dataset '{dataset_key}'."
            )
        entries[entry.label] = entry
    return DatasetState(key=dataset_key, entries_by_label=entries)


# Initialize dataset cache/state
_DATASET_CACHE: Dict[str, DatasetState] = {}
_DEFAULT_DATASET_KEY = os.getenv("DATASET_KEY", _first_key(DATASETS))
if not _DEFAULT_DATASET_KEY:
    raise KeyError("[loading] No dataset entries available in data_key.json.")
_ACTIVE_DATASET_STATE = _load_dataset_state(_DEFAULT_DATASET_KEY)
DATASET_KEY = _ACTIVE_DATASET_STATE.key


def set_dataset_key(dataset_key: str) -> DatasetState:
    """Switch the active dataset context used by helper functions."""
    global _ACTIVE_DATASET_STATE, DATASET_KEY
    if dataset_key == _ACTIVE_DATASET_STATE.key:
        return _ACTIVE_DATASET_STATE
    state = _DATASET_CACHE.get(dataset_key)
    if state is None:
        state = _load_dataset_state(dataset_key)
        _DATASET_CACHE[dataset_key] = state
    _ACTIVE_DATASET_STATE = state
    DATASET_KEY = state.key
    return state


def _ensure_dataset_state(dataset_key: str | None = None) -> DatasetState:
    if dataset_key and dataset_key != _ACTIVE_DATASET_STATE.key:
        return set_dataset_key(dataset_key)
    return _ACTIVE_DATASET_STATE


WEIGHTS_KEY = os.getenv("WEIGHTS_KEY", _first_key(WEIGHT_SETS))
if not WEIGHTS_KEY or WEIGHTS_KEY not in WEIGHT_SETS:
    raise KeyError(f"[loading] Unknown weights key: {WEIGHTS_KEY or 'undefined'}")
MODEL_KEY = WEIGHT_SETS[WEIGHTS_KEY]


def img_path(alt: Any, img: int, dataset_key: str | None = None) -> ImageInstance:
    """Resolve a dataset image to an on-disk path and metadata."""
    state = _ensure_dataset_state(dataset_key)
    label = _normalize_label(alt)
    entry = state.entries_by_label.get(label)
    if entry is None:
        raise SystemExit(
            f"[loading(img_path):warn0] Label '{label}' is not registered for dataset '{state.key}'."
        )
    filename = entry.build_filename(img)
    return ImageInstance(entry=entry, filename=filename)


def weights_path(key: str) -> List[str]:
    key = key.strip()
    if not key:
        raise ValueError("[loading(weights_path)] Empty weight key provided.")

    for folder_name, models in MODEL_KEY.items():
        if key in models:
            hub_entry, file_name, data_name = models[key]
            ckpt = WEIGHT_ROOT / folder_name / file_name
            return [hub_entry, ckpt.as_posix(), data_name]
    raise KeyError(f"[loading(weights_path)] Weight key '{key}' not found in registry '{WEIGHTS_KEY}'.")


def frame_token(imgAlt: Any, imgIndex: Any, dataset_key: str | None = None) -> str:
    """
    Resolve the frame token using the actual image filename when available.
    Falls back to zero-padded index formatting.
    """
    image_spec = img_path(imgAlt, imgIndex, dataset_key=dataset_key)
    return _extract_frame_token(image_spec.filename, imgIndex)


def file_prefix(imgAlt: Any, imgIndex: Any, dataset_key: str | None = None) -> str:
    label = _sanitize_token(_normalize_label(imgAlt))
    token = frame_token(imgAlt, imgIndex, dataset_key=dataset_key)
    return f"{label}_{token}"


def normalize_group_value(value: Any) -> str:
    """Public wrapper so other modules can reuse label normalisation."""
    return _normalize_label(value)


def sanitize_group_token(value: Any) -> str:
    """Return a filesystem-safe token for a dataset group value."""
    return _sanitize_token(_normalize_label(value))


"""Image file I/O and CLI helpers"""


def parse_pair(s: str) -> Tuple[int, str]:
    alt_s, frm_s = s.split(".", 1)
    alt = int(re.sub(r"\D", "", alt_s))
    frame = re.sub(r"\D", "", frm_s).zfill(4)
    if not frame:
        raise SystemExit("[loading(parse_pair):warn1] empty frame")
    return alt, frame


def find_image(img_root: Path, alt: int, frame: str) -> Path:
    for ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"):
        hits = list(img_root.glob(f"**/*_{alt}_{frame}.{ext}"))
        if hits:
            return hits[0]
    raise SystemExit(f"[loading(find_image):warn2] No image for alt={alt}, frame={frame} under {img_root}")


def load_image(path: Path | str) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    return transforms.ToTensor()(im)


def images_regex(root: Path, regex: str, exts: Iterable[str]) -> Dict[str, Path]:
    rx = re.compile(regex, re.IGNORECASE)
    exts = tuple(exts)
    out: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        m = rx.match(str(p).replace("\\", "/"))
        if not m:
            continue
        alt = m.group("alt")
        frame = m.group("frame")
        key = f"{int(alt)}.{frame}"
        out[key] = p
    if not out:
        raise SystemExit(f"[loading:warn3] No images matched under {root}")
    return out


def enumerate_pairs(keys: List[str], a: str = None, b: str = None) -> List[Tuple[str, str]]:
    key_set = set(keys)
    keys_by_alt: Dict[int, List[str]] = defaultdict(list)
    for key in keys:
        alt_str, frame_str = key.split(".", 1)
        alt_val = int(alt_str)
        keys_by_alt[alt_val].append(key)

    def normalize_target(raw: str, label: str) -> List[str]:
        if raw is None:
            return []
        value = raw.strip()
        if not value:
            return []
        if "." in value:
            alt, frame = parse_pair(value)
            key = f"{alt}.{frame}"
            if key not in key_set:
                raise SystemExit(f"[loading:warn3] No image matched for {label}={value}")
            return [key]
        alt_digits = re.sub(r"\D", "", value)
        if not alt_digits:
            raise SystemExit(f"[loading:warn3] Invalid ALT value for {label}: {value}")
        alt_val = int(alt_digits)
        if alt_val not in keys_by_alt:
            raise SystemExit(f"[loading:warn3] No images matched ALT={alt_val} for {label}")
        return list(keys_by_alt[alt_val])

    list_a = normalize_target(a, "--pair-a") or list(keys)
    list_b = normalize_target(b, "--pair-b") or list(keys)

    pairs: List[Tuple[str, str]] = []
    for key_a in list_a:
        for key_b in list_b:
            if key_a == key_b:
                continue
            pairs.append((key_a, key_b))
    return pairs


def save_json(out_dir: Path, stub: str, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stub}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def prepare_run_context(args, weight_groups: Dict[str, List[str]], all_weight_keys: List[str]):
    key2path = images_regex(IMG_ROOT, args.regex, args.exts)
    keys = sorted(key2path.keys(), key=lambda s: (int(s.split('.')[0]), s.split('.')[1]))
    pairs = enumerate_pairs(keys, args.pair_a, args.pair_b)
    print(f"[images] total={len(keys)}  pairs_to_run={len(pairs)}")

    selected_weight_keys = (
        all_weight_keys if args.all_weights
        else weight_groups[args.group] if args.group
        else args.weights
    )
    all_weight_key_set = set(all_weight_keys)
    resolved_weights = []
    dataset_type = None
    for weight_name in selected_weight_keys:
        if weight_name not in all_weight_key_set:
            raise SystemExit(f"Unknown weight key: {weight_name}")
        hub_entry, weight_path_str, dataset_type = weights_path(weight_name)
        weight_path = Path(weight_path_str)
        if not weight_path.is_file():
            raise SystemExit(f"[weight] not found for {weight_name}: {weight_path}")
        resolved_weights.append((weight_name, hub_entry, weight_path))
    print(f"[weights] selected={len(resolved_weights)} -> {[w[0] for w in resolved_weights]}")
    return key2path, pairs, resolved_weights, dataset_type


def save_match_result(
    args,
    weight_name: str,
    hub_entry: str,
    weight_path: Path,
    pair_a: str,
    pair_b: str,
    image_a: Path,
    image_b: Path,
    cosine: float,
    time_ms: Dict[str, float],
    patch: Dict | None,
) -> Path:
    model_root = f"/workspace/weights/{weight_path.name}"
    meta = dict(
        repo_dir=str(REPO_DIR),
        img_root=str(IMG_ROOT),
        embed_root=str(EMBED_ROOT),
        match_root=str(MATCH_ROOT),
        model_root=model_root,
        hub_model=hub_entry,
        device=args.device,
        image_size=int(args.image_size),
    )
    payload = dict(
        meta=meta,
        image_a=str(image_a),
        image_b=str(image_b),
        weight=weight_name,
        cosine=cosine,
        time_ms=dict(
            forward_a=float(time_ms["forward_a"]),
            forward_b=float(time_ms["forward_b"]),
            total=float(time_ms["total"]),
        ),
    )
    payload["advanced_settings"] = dict(
        match_threshold=float(args.match_th),
        max_features=int(args.max_features),
        keypoint_threshold=float(args.keypoint_th),
        line_threshold=float(args.line_th),
        matching_mode="mutual_knn_k1_unique",
    )
    if patch is not None:
        payload["patch"] = patch

    alt_id, frame_id = pair_a.split('.')
    out_dir = MATCH_ROOT / f"{weight_name}_{alt_id}_{frame_id}"
    out_name = f"{weight_name}_{pair_a}_{pair_b}"
    out_path = save_json(out_dir, out_name, payload)
    print(f"[saved] {out_path}")
    return out_path
