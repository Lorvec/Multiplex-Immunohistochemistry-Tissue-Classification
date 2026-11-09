# MIHIC Dataset Setup

Download MIHIC from the official Zenodo record:

https://doi.org/10.5281/zenodo.10065510

Extract the dataset outside the Git repository.

The pipeline searches for a directory containing the following three folders:

```text
MIHIC_ROOT/
    train/
    val/
    test/
```

Each split must contain the seven class directories used by the released MIHIC dataset:

```text
Immune cells/
Necrosis/
Other/
Stroma/
Tumor/
alveoli/
background/
```

Expected split totals are:

```text
Training:   195,001
Validation:  49,260
Test:        65,437
Total:      309,698
```

The code performs a strict audit before training and stops if the expected class structure or counts do not match.

The dataset is not redistributed by this repository.
