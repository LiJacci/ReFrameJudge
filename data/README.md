# Data Directory

This repository should only track lightweight metadata, schemas, sample annotations, and tiny demonstration assets.

Do not commit full image datasets, generated image batches, model checkpoints, or downloaded archives.

Recommended layout:

```text
data/
  raw/                 # Local external datasets. Ignored by git.
  external/            # Optional symlinks or extracted third-party datasets. Ignored by git.
  cache/               # Temporary preprocessing cache. Ignored by git.
  metadata/            # Dataset manifests and source metadata.
  pairs/
    images/
      source/          # Pair source images. Ignored by git except .gitkeep.
      edit/            # Pair edited images. Ignored by git except .gitkeep.
    annotations/       # JSONL annotation files and small examples.
```

For public releases, upload large images and model outputs to a dataset host such as Hugging Face Datasets, Zenodo, or cloud storage, then keep only metadata and download scripts in this repository.

