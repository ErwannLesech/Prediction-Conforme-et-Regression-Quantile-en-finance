"""src/download_data.py
Interface simplifiée pour télécharger les datasets.

Utilisation:
    python -m src.download_data --task classification
    python -m src.download_data --task regression
"""

from __future__ import annotations

import argparse
import sys
from .data_loading import download_classification_data, download_regression_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download datasets for the project")
    parser.add_argument("--task", choices=["classification", "regression"], required=True,
                        help="Which dataset to download: classification or regression")
    args = parser.parse_args(argv)

    try:
        if args.task == "classification":
            download_classification_data()
        elif args.task == "regression":
            download_regression_data()
        print(f"Dataset {args.task} téléchargé avec succès!")
        return 0
    except Exception as e:
        print(f"Erreur pendant le téléchargement: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())