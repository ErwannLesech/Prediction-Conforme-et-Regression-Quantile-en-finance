"""Module pour le chargement et la validation des jeux de données.

Ce module contient les fonctions pour télécharger et charger les datasets
depuis Kaggle, ainsi que les fonctions de validation de base.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    """Retourne le répertoire racine du projet."""
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: Path) -> None:
    """Crée un répertoire s'il n'existe pas."""
    p.mkdir(parents=True, exist_ok=True)


def download_classification_data(dest_dir: Optional[str] = None) -> Path:
    """Télécharge le dataset de classification depuis Kaggle.
    
    Args:
        dest_dir: Répertoire de destination (par défaut: data/raw/)
        
    Returns:
        Path: Chemin vers le répertoire contenant les données
        
    Raises:
        RuntimeError: Si kagglehub n'est pas installé ou configuré
    """
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub est requis pour télécharger les datasets. "
            "Installez-le avec: pip install kagglehub"
        ) from e

    dataset_id = "agewerc/corporate-credit-rating"
    root = _project_root()
    out_dir = Path(dest_dir) if dest_dir else root / "data" / "raw"
    _ensure_dir(out_dir)

    print(f"Téléchargement du dataset de classification: {dataset_id}")
    downloaded_path = kagglehub.dataset_download(dataset_id)
    
    # Gérer les différents formats de retour
    downloaded = Path(downloaded_path)
    if downloaded.is_file() and downloaded.suffix == ".zip":
        print(f"Extraction vers {out_dir}")
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(out_dir)
    elif downloaded.is_file():
        dest_file = out_dir / "corporate_credit_rating.csv"
        shutil.copy2(downloaded, dest_file)
    elif downloaded.is_dir():
        for item in downloaded.iterdir():
            target = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    print(f"Données de classification disponibles dans: {out_dir}")
    return out_dir


def download_regression_data(dest_dir: Optional[str] = None) -> Path:
    """Télécharge le dataset de régression depuis Kaggle.
    
    Args:
        dest_dir: Répertoire de destination (par défaut: data/raw/)
        
    Returns:
        Path: Chemin vers le répertoire contenant les données
        
    Raises:
        RuntimeError: Si kagglehub n'est pas installé ou configuré
    """
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub est requis pour télécharger les datasets. "
            "Installez-le avec: pip install kagglehub"
        ) from e

    dataset_id = "lorenzozoppelletto/financial-risk-for-loan-approval"
    root = _project_root()
    out_dir = Path(dest_dir) if dest_dir else root / "data" / "raw"
    _ensure_dir(out_dir)

    print(f"Téléchargement du dataset de régression: {dataset_id}")
    downloaded_path = kagglehub.dataset_download(dataset_id)
    
    downloaded = Path(downloaded_path)
    if downloaded.is_file() and downloaded.suffix == ".zip":
        print(f"Extraction vers {out_dir}")
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(out_dir)
    elif downloaded.is_file():
        # Copier le fichier principal (probablement Loan.csv)
        dest_file = out_dir / downloaded.name
        shutil.copy2(downloaded, dest_file)
    elif downloaded.is_dir():
        for item in downloaded.iterdir():
            target = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    print(f"Données de régression disponibles dans: {out_dir}")
    return out_dir


def load_classification_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Charge le dataset de classification.
    
    Args:
        data_dir: Répertoire contenant les données (par défaut: data/raw/)
        
    Returns:
        DataFrame contenant les données de classification
        
    Raises:
        FileNotFoundError: Si le fichier de données n'est pas trouvé
    """
    root = _project_root()
    data_path = Path(data_dir) if data_dir else root / "data" / "raw"
    
    # Chercher spécifiquement Loan.csv ou tout autre CSV
    loan_file = data_path / "corporate_rating.csv"
    if loan_file.exists():
        csv_file = loan_file
    else:
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Aucun fichier CSV trouvé dans {data_path}. "
                "Avez-vous téléchargé les données avec download_regression_data() ?"
            )
        csv_file = csv_files[0]
    
    print(f"Chargement du dataset de classification: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    return df


def load_regression_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Charge le dataset de régression.
    
    Args:
        data_dir: Répertoire contenant les données (par défaut: data/raw/)
        
    Returns:
        DataFrame contenant les données de régression
        
    Raises:
        FileNotFoundError: Si le fichier de données n'est pas trouvé
    """
    root = _project_root()
    data_path = Path(data_dir) if data_dir else root / "data" / "raw"
    
    # Chercher spécifiquement Loan.csv ou tout autre CSV
    loan_file = data_path / "Loan.csv"
    if loan_file.exists():
        csv_file = loan_file
    else:
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Aucun fichier CSV trouvé dans {data_path}. "
                "Avez-vous téléchargé les données avec download_regression_data() ?"
            )
        csv_file = csv_files[0]
    
    print(f"Chargement du dataset de régression: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    return df


def validate_data(df: pd.DataFrame, task_type: str = "classification") -> dict:
    """Valide un dataset et retourne des informations de base.
    
    Args:
        df: DataFrame à valider
        task_type: Type de tâche ("classification" ou "regression")
        
    Returns:
        Dictionnaire contenant les informations de validation
    """
    info = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().sum(),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist()
    }
    
    if task_type == "classification":
        # Supposer que la dernière colonne est la target pour la classification
        if len(df.columns) > 0:
            target_col = df.columns[-1]
            info["target_column"] = target_col
            info["target_classes"] = df[target_col].value_counts().to_dict()
    
    return info