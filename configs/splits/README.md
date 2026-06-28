# Dataset Split Lists

This repo supports reproducible train/val splits for image-folder style datasets via text files referenced from config:

```yaml
data:
  splits:
    train: ./configs/splits/celeba_train.txt
    val:   ./configs/splits/celeba_val.txt
```

## File format

- Plain text, one image path per line.
- Lines can be:
  - relative paths (resolved against the dataset root), or
  - absolute paths.
- Empty lines are ignored.
- Lines starting with `#` are treated as comments.

## Path resolution rules

- For `dataset: imagefolder`, relative paths are resolved against `data.root`.
- For `dataset: celeba`, relative paths are resolved against the detected CelebA image directory:
  - `data.root/img_align_celeba/` (preferred), else
  - `data.root/celeba/`, else
  - `data.root/`

## Validation

If any listed file does not exist, dataset construction fails fast with a preview of missing paths.

