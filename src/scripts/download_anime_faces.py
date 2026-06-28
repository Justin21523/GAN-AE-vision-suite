"""
Download and prepare a large anime face image dataset for GAN training.

Default source: Hugging Face dataset repo `huggan/anime-faces`.

This script:
- downloads `data.zip` from the HF dataset repo
- extracts it into `--out-root` (default: ./data/anime_faces)
- leaves images under `--out-root/images` for use with `data.dataset: imagefolder`

Example:
  python -m src.scripts.download_anime_faces --out-root ./data/anime_faces
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def _count_images(root: Path) -> int:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff")
    n = 0
    for ext in exts:
        n += len(list(root.rglob(ext)))
        n += len(list(root.rglob(ext.upper())))
    return n


def _find_best_image_dir(out_root: Path) -> Path | None:
    """
    Find the directory under `out_root` that contains the most images.

    Some sources extract to e.g. `out_root/data/*.png` or `out_root/images/*.png`.
    """
    top_level_has_images = any(
        p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        for p in out_root.iterdir()
    )
    if top_level_has_images:
        return out_root

    subdirs = [p for p in out_root.iterdir() if p.is_dir()]
    best: tuple[int, Path] | None = None
    for d in subdirs:
        n = _count_images(d)
        if n <= 0:
            continue
        if best is None or n > best[0]:
            best = (n, d)
    return best[1] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", type=str, default="huggan/anime-faces")
    ap.add_argument("--filename", type=str, default="data.zip")
    ap.add_argument("--out-root", type=str, default="/mnt/data/datasets/gan-ae-vision-suite/data/anime_faces")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output folder already contains images.",
    )
    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="If >0, create a small subset folder with N images (for quick smoke tests).",
    )
    ap.add_argument(
        "--subset-dir",
        type=str,
        default=None,
        help="Where to write subset images (default: <out-root>/subset_<N>/images).",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_images = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)

    existing_any = _count_images(out_root)
    if existing_any > 0 and not args.force:
        # Already extracted or otherwise present; just normalize the expected folder structure.
        if out_images.exists() and _count_images(out_images) == 0:
            shutil.rmtree(out_images)
        if not out_images.exists():
            best = _find_best_image_dir(out_root)
            if best is None:
                raise SystemExit(f"Could not locate any image directory under {out_root}.")
            if best == out_root:
                out_images.mkdir(parents=True, exist_ok=True)
                for p in list(out_root.iterdir()):
                    if p.is_file() and p.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".bmp",
                        ".tiff",
                    }:
                        shutil.move(str(p), str(out_images / p.name))
                print(f"Collected top-level images into: {out_images}")
            else:
                shutil.move(str(best), str(out_images))
                print(f"Moved image folder from {best} -> {out_images}")
        print(f"Found existing dataset under: {out_root} ({existing_any} image files total)")
        print(f"Images ready at: {out_images} ({_count_images(out_images)} files)")
    else:
        if out_images.exists() and args.force:
            shutil.rmtree(out_images)

        print(f"Downloading {args.repo_id}:{args.filename} ...")
        zip_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type="dataset",
            )
        )
        print(f"Downloaded to cache: {zip_path}")

        print(f"Extracting into: {out_root}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_root)

        n = _count_images(out_root)
        if n == 0:
            raise SystemExit(
                f"Extraction finished but no images found under {out_root}. "
                "Check dataset structure or choose a different filename."
            )

        # Normalize: ensure we always have <out-root>/images/*.png for configs.
        if not out_images.exists():
            best = _find_best_image_dir(out_root)
            if best is None:
                raise SystemExit(f"Could not locate any image directory under {out_root}.")
            if best == out_root:
                # Images extracted directly into out_root; keep them but provide a stable folder.
                out_images.mkdir(parents=True, exist_ok=True)
                for p in list(out_root.iterdir()):
                    if p.is_file() and p.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".bmp",
                        ".tiff",
                    }:
                        shutil.move(str(p), str(out_images / p.name))
                print(f"Collected top-level images into: {out_images}")
            else:
                shutil.move(str(best), str(out_images))
                print(f"Moved image folder from {best} -> {out_images}")

        print(f"Done. Images ready at: {out_images} ({_count_images(out_images)} files)")

    if int(args.subset) > 0:
        subset_n = int(args.subset)
        subset_root = (
            Path(args.subset_dir).resolve()
            if args.subset_dir
            else (out_root / f"subset_{subset_n}" / "images").resolve()
        )
        subset_root.mkdir(parents=True, exist_ok=True)

        imgs = []
        for ext in ("png", "jpg", "jpeg", "webp", "bmp", "tiff"):
            imgs.extend(list(out_images.rglob(f"*.{ext}")))
            imgs.extend(list(out_images.rglob(f"*.{ext.upper()}")))
        imgs.sort()
        if not imgs:
            raise SystemExit(f"No images found to subset under: {out_images}")

        print(f"Creating subset: {subset_root} (N={subset_n})")
        for src in imgs[:subset_n]:
            dst = subset_root / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"Subset ready at: {subset_root} ({_count_images(subset_root)} files)")


if __name__ == "__main__":
    main()
