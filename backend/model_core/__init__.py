"""
algoforge's model architectures and training-loop core — deliberately free of any dependency on
the rest of the backend (FastAPI, SQLAlchemy, Celery, the `data`/`webhooks` packages) so it can
be installed standalone, e.g. into a Colab runtime via:

    pip install "git+https://github.com/Naradice/algoforge.git@<commit>#subdirectory=backend/model_core"

This is what makes it possible for a Colab-executed training run (see
../model/colab_trainer.py and ../model/notebook_export.py) to run the *actual* algoforge model
and training-loop code rather than a hand-copied reimplementation that could drift from it —
both the backend's own celery worker and a Colab notebook import this same package.

Two things stay lazy-imported inside model_core/trainers/dataset.py specifically so importing
this package doesn't require them unless actually used:
    - scikit-learn (only for OHLCWindowDataset's token_level="cluster")
    - model_core.trainers.preprocessing, which itself needs the separate `finance_client`
      package (only for the `preprocessing` indicators/clustering recipe option)
Neither is a dependency of this package's default (Colab-supported) path.
"""
