"""TF-IDF/cosine vs. semantic retrieval on BEIR SciFact."""

import os

# Windows without Developer Mode can't symlink in the HF cache; the fallback works, just say less.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
