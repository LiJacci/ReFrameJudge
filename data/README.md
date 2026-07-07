# Data Directory

This repository tracks lightweight metadata, schemas, annotations, and the small FCDB 5k crop subset used by the current baselines.

Do not commit full image datasets, generated image batches, model checkpoints, embedding caches, or downloaded archives.

Recommended layout:

```text
data/
  raw/                 # Local external datasets. Ignored by git.
  external/            # Optional symlinks or extracted third-party datasets.
  cache/               # Temporary preprocessing cache. Ignored by git.
  metadata/            # Dataset manifests and source metadata.
  pairs/
    images/
      source/          # Pair source images for the tracked FCDB 5k crop subset.
      edit/            # Pair edited images for the tracked FCDB 5k crop subset.
    annotations/       # JSONL annotation files and small examples.
```

For public releases or larger generated-image datasets, upload large images and model outputs to a dataset host such as Hugging Face Datasets, Zenodo, or cloud storage, then keep only metadata and download scripts in this repository.
