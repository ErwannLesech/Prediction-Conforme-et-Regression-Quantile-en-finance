# Preprocessing des données
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def clean_data(df, target_column=None):
    print("Nettoyage des données")
    df_clean = df.copy()
    
    # Supprimer les doublons
    initial = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    if df_clean.shape[0] < initial:
        print("Doublons supprimés:", initial - df_clean.shape[0])
    
    # Valeurs manquantes
    if df_clean.isnull().sum().sum() > 0:
        print("Suppression des valeurs manquantes")
        df_clean = df_clean.dropna()
    
    print("Shape finale:", df_clean.shape)
    return df_clean


def encode_categorical_features(df, categorical_columns=None, encoding_type="onehot"):
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    df_encoded = df.copy()
    encoders = {}
    
    if not categorical_columns:
        return df_encoded, encoders
    
    print("Encodage des variables catégorielles:", len(categorical_columns))
    
    if encoding_type == "onehot":
        for col in categorical_columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                encoders[col] = f"onehot_{col}"
                
    elif encoding_type == "label":
        for col in categorical_columns:
            if col in df_encoded.columns:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                encoders[col] = encoder
    
    return df_encoded, encoders


def normalize_features(df, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_scaled = df.copy()
    
    if not numeric_columns:
        return df_scaled, None
    
    print("Normalisation de", len(numeric_columns), "variables")
    
    scaler = StandardScaler()
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
    
    return df_scaled, scaler


def group_credit_ratings(rating_series, min_samples=10):
    # Regroupement des ratings
    rating_mapping = {
        'AAA': 'IG_HIGH',
        'AA': 'IG_HIGH',
        'A': 'IG_MED',
        'BBB': 'IG_LOW',
        'BB': 'SPEC_HIGH',
        'B': 'SPEC_MED',
        'CCC': 'SPEC_LOW',
        'CC': 'SPEC_LOW',
        'C': 'SPEC_LOW',
        'D': 'SPEC_LOW'
    }
    
    grouped = rating_series.map(rating_mapping)
    print("Distribution après regroupement:")
    print(grouped.value_counts())
    
    return grouped


def prepare_classification_data(df, target_column, test_size=0.2, random_state=42, group_ratings=True):
    print("Préparation classification")
    print("Target:", target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()
    
    if group_ratings and target_column.lower() == 'rating':
        y = group_credit_ratings(y)
    
    print("Nombre de classes:", y.nunique())
    
    # Check pour stratification
    min_class_size = y.value_counts().min()
    stratify = None if min_class_size < 2 else y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print("Train:", X_train.shape, "Test:", X_test.shape)
    
    return X_train, X_test, y_train, y_test


def prepare_regression_data(df, target_column, test_size=0.2, random_state=42):
    print("Préparation régression")
    print("Target:", target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("Train:", X_train.shape, "Test:", X_test.shape)
    
    return X_train, X_test, y_train, y_test


def save_processed_data(data, filename, data_dir=None):
    root = Path(__file__).resolve().parents[1]
    save_dir = Path(data_dir) if data_dir else root / "data" / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        filepath = save_dir / f"{filename}.csv"
        data.to_csv(filepath, index=False)
        print("Sauvegardé:", filepath)
        return filepath
    
    elif isinstance(data, tuple) and len(data) == 4:
        X_train, X_test, y_train, y_test = data
        
        for name, df in [("X_train", X_train), ("X_test", X_test), 
                        ("y_train", y_train), ("y_test", y_test)]:
            filepath = save_dir / f"{filename}_{name}.csv"
            df.to_csv(filepath, index=False)
        
        print("Données sauvegardées dans:", save_dir)
        return save_dir
    
    else:
        raise ValueError("Format non supporté")