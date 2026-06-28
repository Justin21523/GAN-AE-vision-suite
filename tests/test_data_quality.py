"""
Data quality helper tests.

Keep these CPU/lightweight by using tiny temporary image folders.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.quality import collect_image_paths, scan_images


def test_scan_images_counts(tmp_path: Path) -> None:
    # Create 2 valid images and 1 invalid "image" file.
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(tmp_path / "a.png")
    Image.new("RGB", (9, 7), color=(1, 2, 3)).save(tmp_path / "sub" / "b.png")
    (tmp_path / "bad.png").write_bytes(b"not an image")

    paths = collect_image_paths(tmp_path)
    result = scan_images(paths, max_bad_paths=10)
    assert result.total_files == 3
    assert result.ok_files == 2
    assert result.bad_files == 1

