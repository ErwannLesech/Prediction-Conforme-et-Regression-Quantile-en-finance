"""Module de préparation et préprocessing des données.

Ce module contient les fonctions pour nettoyer, encoder et normaliser
les données avant l'entraînement des modèles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """Nettoie les données de base (valeurs manquantes, doublons).
    
    Args:
        df: DataFrame à nettoyer
        target_column: Nom de la colonne cible (optionnel)
        
    Returns:
        DataFrame nettoyé
    """
    print(f"Nettoyage des données - Shape initiale: {df.shape}")
    
    # Copie pour éviter les modifications sur l'original
    df_clean = df.copy()
    
    # Supprimer les doublons
    initial_shape = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    if df_clean.shape[0] < initial_shape:
        print(f"Suppression de {initial_shape - df_clean.shape[0]} doublons")
    
    # Gérer les valeurs manquantes
    missing_counts = df_clean.isnull().sum()
    if missing_counts.sum() > 0:
        print("Valeurs manquantes détectées:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} ({count/len(df_clean)*100:.1f}%)")
        
        # Stratégie simple: suppression des lignes avec des valeurs manquantes
        # Dans un projet réel, on pourrait implémenter des stratégies plus sophistiquées
        df_clean = df_clean.dropna()
        print(f"Après suppression des valeurs manquantes: {df_clean.shape}")
    
    print(f"Shape finale après nettoyage: {df_clean.shape}")
    return df_clean


def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_columns: Optional[list] = None,
    encoding_type: str = "onehot"
) -> Tuple[pd.DataFrame, dict]:
    """Encode les variables catégorielles.
    
    Args:
        df: DataFrame contenant les données
        categorical_columns: Liste des colonnes catégorielles (détection auto si None)
        encoding_type: Type d'encodage ("onehot", "label")
        
    Returns:
        Tuple contenant le DataFrame encodé et les encodeurs utilisés
    """
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    df_encoded = df.copy()
    encoders = {}
    
    if not categorical_columns:
        print("Aucune variable catégorielle détectée")
        return df_encoded, encoders
    
    print(f"Encodage de {len(categorical_columns)} variables catégorielles: {categorical_columns}")
    
    if encoding_type == "onehot":
        for col in categorical_columns:
            if col in df_encoded.columns:
                # One-hot encoding avec pandas pour plus de simplicité
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                encoders[col] = f"pd.get_dummies_{col}"
                
    elif encoding_type == "label":
        for col in categorical_columns:
            if col in df_encoded.columns:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                encoders[col] = encoder
    
    print(f"Shape après encodage: {df_encoded.shape}")
    return df_encoded, encoders


def normalize_features(
    df: pd.DataFrame, 
    numeric_columns: Optional[list] = None,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, object]:
    """Normalise les variables numériques.
    
    Args:
        df: DataFrame contenant les données
        numeric_columns: Liste des colonnes numériques (détection auto si None)
        scaler_type: Type de normalisation ("standard", "minmax")
        
    Returns:
        Tuple contenant le DataFrame normalisé et le scaler utilisé
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_scaled = df.copy()
    
    if not numeric_columns:
        print("Aucune variable numérique détectée")
        return df_scaled, None
    
    print(f"Normalisation de {len(numeric_columns)} variables numériques")
    
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
    
    return df_scaled, scaler


def group_credit_ratings(rating_series: pd.Series, min_samples: int = 10) -> pd.Series:
    """Regroupe les ratings de crédit pour équilibrer les classes.
    
    Stratégie de regroupement basée sur la hiérarchie des ratings :
    - Investment Grade: AAA, AA, A, BBB → IG_HIGH (AAA, AA), IG_MED (A), IG_LOW (BBB)
    - Speculative Grade: BB, B, CCC, CC, C, D → SPEC_HIGH (BB), SPEC_MED (B), SPEC_LOW (CCC, CC, C, D)
    
    Args:
        rating_series: Série contenant les ratings originaux
        min_samples: Nombre minimum d'échantillons par classe
        
    Returns:
        Série avec les ratings regroupés
    """
    print("Regroupement des ratings de crédit...")
    print("Distribution originale:")
    original_counts = rating_series.value_counts().sort_index()
    print(original_counts)
    
    # Mapping de regroupement
    rating_mapping = {
        # Investment Grade High (très haute qualité)
        'AAA': 'IG_HIGH',
        'AA': 'IG_HIGH',
        
        # Investment Grade Medium (haute qualité)
        'A': 'IG_MED',
        
        # Investment Grade Low (qualité satisfaisante)
        'BBB': 'IG_LOW',
        
        # Speculative High (modérément spéculatif)
        'BB': 'SPEC_HIGH',
        
        # Speculative Medium (spéculatif)
        'B': 'SPEC_MED',
        
        # Speculative Low (très spéculatif/défaut)
        'CCC': 'SPEC_LOW',
        'CC': 'SPEC_LOW',
        'C': 'SPEC_LOW',
        'D': 'SPEC_LOW'
    }
    
    # Application du mapping
    grouped_ratings = rating_series.map(rating_mapping)
    
    print("\nDistribution après regroupement:")
    grouped_counts = grouped_ratings.value_counts().sort_index()
    print(grouped_counts)
    
    # Vérification du minimum d'échantillons
    min_count = grouped_counts.min()
    if min_count < min_samples:
        print(f"\nAttention: La classe avec le moins d'échantillons a {min_count} éléments")
        print("Considérez d'augmenter la collecte de données ou d'ajuster le regroupement")
    
    return grouped_ratings


def prepare_classification_data(
    df: pd.DataFrame, 
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    group_ratings: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prépare les données pour une tâche de classification.
    
    Args:
        df: DataFrame contenant les données
        target_column: Nom de la colonne cible
        test_size: Proportion des données de test
        random_state: Graine aléatoire
        group_ratings: Si True, regroupe les ratings pour équilibrer les classes
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Préparation des données de classification - Target: {target_column}")
    
    # Séparer features et target
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()
    
    # Regroupement optionnel des ratings
    if group_ratings and target_column.lower() == 'rating':
        y = group_credit_ratings(y)
    
    print(f"Nombre de classes: {y.nunique()}")
    print(f"Distribution des classes:")
    print(y.value_counts(normalize=True))
    
    # Vérification pour la stratification
    min_class_size = y.value_counts().min()
    if min_class_size < 2:
        print(f"\nAttention: Certaines classes ont moins de 2 échantillons (min: {min_class_size})")
        print("Utilisation d'un split sans stratification")
        stratify = None
    else:
        stratify = y
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"Données d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def prepare_regression_data(
    df: pd.DataFrame,
    target_column: str, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prépare les données pour une tâche de régression.
    
    Args:
        df: DataFrame contenant les données
        target_column: Nom de la colonne cible
        test_size: Proportion des données de test
        random_state: Graine aléatoire
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Préparation des données de régression - Target: {target_column}")
    
    # Séparer features et target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Statistiques de la variable cible:")
    print(y.describe())
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Données d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(
    data: Union[pd.DataFrame, Tuple], 
    filename: str, 
    data_dir: Optional[str] = None
) -> Path:
    """Sauvegarde les données préprocessées.
    
    Args:
        data: DataFrame ou tuple de DataFrames à sauvegarder
        filename: Nom du fichier (sans extension)
        data_dir: Répertoire de destination (par défaut: data/processed/)
        
    Returns:
        Path vers le fichier sauvegardé
    """
    root = Path(__file__).resolve().parents[1]
    save_dir = Path(data_dir) if data_dir else root / "data" / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        filepath = save_dir / f"{filename}.csv"
        data.to_csv(filepath, index=False)
        print(f"Données sauvegardées: {filepath}")
        return filepath
    
    elif isinstance(data, tuple) and len(data) == 4:
        # Supposer qu'il s'agit de X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = data
        
        filepaths = []
        for name, df in [("X_train", X_train), ("X_test", X_test), 
                        ("y_train", y_train), ("y_test", y_test)]:
            filepath = save_dir / f"{filename}_{name}.csv"
            df.to_csv(filepath, index=False)
            filepaths.append(filepath)
        
        print(f"Données d'entraînement/test sauvegardées dans: {save_dir}")
        return save_dir
    
    else:
        raise ValueError("Format de données non supporté pour la sauvegarde")