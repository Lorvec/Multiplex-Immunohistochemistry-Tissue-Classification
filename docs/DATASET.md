# MIHIC dataset setup

## Official source

- Dataset: **MIHIC: A multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification**
- Zenodo DOI: https://doi.org/10.5281/zenodo.10065510
- Archive: `MIHIC_dataset.zip` (7.6 GB)
- Archive MD5 reported by Zenodo: `bd718137f70427d530947862332f4cdd`

No image data are included in this repository.

## Expected directory layout

Pass `--data-root` the directory containing the three official split folders:

```text
<data-root>/
├── train/
│   ├── Immune cells/
│   ├── Necrosis/
│   ├── Other/
│   ├── Stroma/
│   ├── Tumor/
│   ├── alveoli/
│   └── background/
├── val/
│   └── <same seven class folders>/
└── test/
    └── <same seven class folders>/
```

Folder capitalization matters because `torchvision.datasets.ImageFolder` determines class indices from lexicographically sorted directory names. The class order in the frozen run was:

```text
Immune cells, Necrosis, Other, Stroma, Tumor, alveoli, background
```

## Expected counts

| Split | Immune cells | Necrosis | Other | Stroma | Tumor | Alveoli | Background | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 6,630 | 14,468 | 37,288 | 25,023 | 91,357 | 7,636 | 12,599 | 195,001 |
| Validation | 1,817 | 1,857 | 11,188 | 5,362 | 24,625 | 1,262 | 3,149 | 49,260 |
| Testing | 2,393 | 7,694 | 12,937 | 7,592 | 27,993 | 2,891 | 3,937 | 65,437 |
| **Overall** | **10,840** | **24,019** | **61,413** | **37,977** | **143,975** | **11,789** | **19,685** | **309,698** |

## Validation

Run the full release check before training:

```bash
python scripts/validate_dataset.py --data-root /path/to/MIHIC
```

This compares the local extraction with `results/manifests/dataset_manifest.csv.gz`. The public manifest contains relative filenames, class/stain/section metadata, and file sizes only. It contains no pixels and no absolute local paths.

Expected portable-manifest SHA-256:

```text
70d8f78674402f157adc21b7480b1458ee434477d41cfff4425ffc155b21d996
```

The digest is calculated from `split`, `relative_path`, `class`, and `file_size_bytes`, in the frozen manifest order.

## Scope of the official partitions

The dataset creators report that patches from the same annotated region are kept in the same split. The release does not provide patient identifiers or a patient-to-split mapping, so this repository cannot verify patient-independent separation. Results therefore concern the official region-grouped patch benchmark.
