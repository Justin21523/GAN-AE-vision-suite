"""
Download a high-resolution (HQ) anime image dataset with a permissive license tag (CC0)
from Hugging Face, and prepare it as an `imagefolder` for GAN training.

Recommended sources (have `license:cc0-1.0` tag on HF at time of writing):
- `alfredplpl/anime-with-caption-cc0` (captioned, 1024px images)
- `umzi/AnimeSet` (1024px images)

This script streams examples and writes resized square images to:
  <out-root>/images/*.png (or jpg)

Examples:
  # Download 50k images, center-crop to square, resize to 512
  python -m src.scripts.download_anime_hq_cc0 \\
    --repo alfredplpl/anime-with-caption-cc0 \\
    --out-root ./data/anime_hq_cc0 \\
    --max-images 50000 \\
    --center-crop --resize 512

  # Face-ish subset via caption keyword filtering (best-effort)
  python -m src.scripts.download_anime_hq_cc0 \\
    --repo alfredplpl/anime-with-caption-cc0 \\
    --out-root ./data/anime_hq_cc0_faces \\
    --max-images 50000 \\
    --filter-any \"face,portrait,close-up,headshot,顔,アップ,ポートレート\" \\
    --center-crop --resize 512
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from PIL import Image


def _find_first_key(d: dict[str, Any], keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in d:
            return k
    return None


def _to_pil(img: Any) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    # datasets.Image in non-streaming may be dict-like; streaming is usually PIL.
    try:
        if hasattr(img, "convert"):
            return img  # type: ignore[return-value]
    except Exception:
        return None
    return None


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)


def _caption_matches(text: str, any_terms: list[str]) -> bool:
    if not any_terms:
        return True
    t = text.lower()
    return any(term.lower() in t for term in any_terms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="alfredplpl/anime-with-caption-cc0")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out-root", type=str, default="/mnt/data/datasets/gan-ae-vision-suite/data/anime_hq_cc0")
    ap.add_argument("--max-images", type=int, default=50000)
    ap.add_argument("--image-key", type=str, default=None, help="Override image column key (default: auto).")
    ap.add_argument(
        "--caption-key",
        type=str,
        default=None,
        help="Override caption/text column key used for filtering (default: auto).",
    )
    ap.add_argument(
        "--filter-any",
        type=str,
        default="",
        help="Comma-separated keywords; keep example if caption contains ANY of them (best-effort).",
    )
    ap.add_argument("--center-crop", action="store_true", help="Center-crop to square before resize.")
    ap.add_argument("--resize", type=int, default=512, help="Resize square size (e.g., 512).")
    ap.add_argument("--format", type=str, choices=["png", "jpg"], default="png")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality when --format=jpg.")
    ap.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every N saved images.",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Stop early if disk free space under out-root falls below this threshold.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_images = out_root / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.jsonl"

    any_terms = [s.strip() for s in str(args.filter_any).split(",") if s.strip()]

    # Resume: pick next index from existing files.
    existing = sorted(out_images.glob(f"*.{args.format}"))
    next_idx = len(existing)

    print(f"Repo: {args.repo} split={args.split}")
    print(f"Out:  {out_images} (resume at idx={next_idx})")
    if any_terms:
        print(f"Filter(any): {any_terms}")
    print(f"Preprocess: center_crop={bool(args.center_crop)} resize={int(args.resize)} format={args.format}")
    if float(args.min_free_gb) > 0:
        print(f"Disk guard: stop if free < {float(args.min_free_gb):.1f} GB")

    ds = load_dataset(args.repo, split=args.split, streaming=True)
    it = iter(ds)

    saved = 0
    scanned = 0
    with open(meta_path, "a", encoding="utf-8") as mf:
        while saved < int(args.max_images):
            if float(args.min_free_gb) > 0:
                free_gb = shutil.disk_usage(out_root).free / (1024**3)
                if free_gb < float(args.min_free_gb):
                    print(f"Stopping: free space {free_gb:.1f} GB < min_free_gb {float(args.min_free_gb):.1f} GB")
                    break
            ex = next(it, None)
            if ex is None:
                break
            scanned += 1

            if args.image_key:
                img_key = args.image_key
            else:
                img_key = _find_first_key(ex, ["image", "img", "png", "jpg"])
            if not img_key:
                continue

            if args.caption_key:
                cap_key = args.caption_key
            else:
                cap_key = _find_first_key(ex, ["prompt", "caption", "text", "phi3_caption", "phi3_caption_ja"])

            caption = ""
            if cap_key and isinstance(ex.get(cap_key), str):
                caption = str(ex.get(cap_key))
            if not _caption_matches(caption, any_terms):
                continue

            img = _to_pil(ex.get(img_key))
            if img is None:
                continue

            img = img.convert("RGB")
            if args.center_crop:
                img = _center_crop_square(img)
            img = _resize(img, int(args.resize))

            out_path = out_images / f"{next_idx:08d}.{args.format}"
            if args.format == "png":
                img.save(out_path, format="PNG", optimize=True)
            else:
                img.save(out_path, format="JPEG", quality=int(args.quality), optimize=True)

            mf.write(
                json.dumps(
                    {
                        "idx": next_idx,
                        "repo": args.repo,
                        "split": args.split,
                        "caption": caption,
                        "path": str(out_path.relative_to(out_root)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            next_idx += 1
            saved += 1

            pe = max(1, int(args.print_every))
            if saved == 1 or (saved % pe == 0):
                print(f"saved={saved} scanned={scanned} last={out_path}")

    print(f"Done. saved={saved} scanned={scanned} out={out_images}")
    print("Next steps:")
    print(f"  python -m src.scripts.make_image_splits --root {out_images} --train-out ./configs/splits/anime_cc0_train.txt --val-out ./configs/splits/anime_cc0_val.txt --val-count 5000 --seed 42")
    print("  (then point your GAN config data.root to this images folder)")


if __name__ == "__main__":
    # Avoid HF progress bars in nohup by default.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
