# Chargement des données
import shutil
import zipfile
from pathlib import Path
import pandas as pd


def _project_root():
    return Path(__file__).resolve().parents[1]


def download_classification_data(dest_dir=None):
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("Installer kagglehub: pip install kagglehub")

    dataset_id = "agewerc/corporate-credit-rating"
    root = _project_root()
    out_dir = Path(dest_dir) if dest_dir else root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Téléchargement des données de classification...")
    downloaded_path = kagglehub.dataset_download(dataset_id)
    
    downloaded = Path(downloaded_path)
    if downloaded.is_file() and downloaded.suffix == ".zip":
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(out_dir)
    elif downloaded.is_file():
        dest_file = out_dir / "corporate_rating.csv"
        shutil.copy2(downloaded, dest_file)
    elif downloaded.is_dir():
        for item in downloaded.iterdir():
            target = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    print("Téléchargement terminé")
    return out_dir


def download_regression_data(dest_dir=None):
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("Installer kagglehub: pip install kagglehub")

    dataset_id = "lorenzozoppelletto/financial-risk-for-loan-approval"
    root = _project_root()
    out_dir = Path(dest_dir) if dest_dir else root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Téléchargement des données de régression...")
    downloaded_path = kagglehub.dataset_download(dataset_id)
    
    downloaded = Path(downloaded_path)
    if downloaded.is_file() and downloaded.suffix == ".zip":
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(out_dir)
    elif downloaded.is_file():
        dest_file = out_dir / downloaded.name
        shutil.copy2(downloaded, dest_file)
    elif downloaded.is_dir():
        for item in downloaded.iterdir():
            target = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    print("Téléchargement terminé")
    return out_dir


def load_classification_data(data_dir=None):
    root = _project_root()
    data_path = Path(data_dir) if data_dir else root / "data" / "raw"
    
    csv_file = data_path / "corporate_rating.csv"
    if not csv_file.exists():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("Aucun fichier CSV trouvé")
        csv_file = csv_files[0]
    
    print("Chargement des données...")
    df = pd.read_csv(csv_file)
    print("Shape:", df.shape)
    
    return df


def load_regression_data(data_dir=None):
    root = _project_root()
    data_path = Path(data_dir) if data_dir else root / "data" / "raw"
    
    csv_file = data_path / "Loan.csv"
    if not csv_file.exists():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("Aucun fichier CSV trouvé")
        csv_file = csv_files[0]
    
    print("Chargement des données...")
    df = pd.read_csv(csv_file)
    print("Shape:", df.shape)
    
    return df