from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            if enc == "utf-8" and "BOM" in str(exc):
                continue
            raise
    raise ValueError(f"Failed to parse JSON: {path}")


def _num(data: Dict[str, Any], key: str) -> float | None:
    val = data.get(key) if isinstance(data, dict) else None
    return float(val) if isinstance(val, (int, float)) else None


def build_imageref_index(data: Dict[str, Any]) -> Dict[str, Dict[str, float | None]]:
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError("Expected 'images' list in label JSON.")

    mapping: Dict[str, Dict[str, float | None]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        ref = item.get("imageref")
        if not isinstance(ref, str) or not ref:
            continue
        pos = item.get("position") if isinstance(item.get("position"), dict) else {}
        att = item.get("attitude") if isinstance(item.get("attitude"), dict) else {}
        mapping[ref] = {
            "x": _num(pos, "x"),
            "y": _num(pos, "y"),
            "z": _num(pos, "z"),
            "latitude": _num(pos, "latitude"),
            "longitude": _num(pos, "longitude"),
            "altitude": _num(pos, "altitude"),
            "yaw": _num(att, "yaw"),
            "pitch": _num(att, "pitch"),
            "roll": _num(att, "roll"),
        }
    return mapping


def merge_indexes(
    base: Dict[str, Dict[str, float | None]],
    extra: Dict[str, Dict[str, float | None]],
    overwrite: bool,
    source: Path,
) -> None:
    for key, value in extra.items():
        if key in base and not overwrite:
            if base[key] != value:
                print(f"[WARN] duplicate imageref (kept existing): {key} from {source}")
            continue
        if key in base and overwrite and base[key] != value:
            print(f"[WARN] duplicate imageref (overwrote): {key} from {source}")
        base[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build imageref -> lat/long/alt/yaw/pitch/roll mapping from label JSON files.",
    )
    parser.add_argument(
        "--json",
        action="append",
        type=Path,
        required=True,
        help="Path to a label JSON file (repeatable).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for the mapping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite duplicate imageref entries when merging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out: Dict[str, Dict[str, float | None]] = {}

    for json_path in args.json:
        if not json_path.exists():
            raise FileNotFoundError(f"Label JSON not found: {json_path}")
        data = _read_json(json_path)
        mapping = build_imageref_index(data)
        merge_indexes(out, mapping, overwrite=bool(args.overwrite), source=json_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] saved {len(out)} entries -> {args.out}")


if __name__ == "__main__":
    main()
