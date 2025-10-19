"""
download_data.py
Script to download UCI Statlog (German Credit Data) dataset and save it as a CSV file in the data/raw/ directory.
"""

import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
statlog = fetch_ucirepo(id=144)

X = statlog.data.features
y = statlog.data.targets

# Fusionner X et y
df = pd.concat([X, y], axis=1)

# Liste des colonnes explicites
columns = [
    "checking_account", "duration_month", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker", "credit_risk"
]

df.columns = columns

# Remapping de la cible pour plus de clarté
df["credit_risk"] = df["credit_risk"].map({1: "good", 2: "bad"})

print("Downloading and processing the German Credit Data dataset...")
print("---------------------")

print("Dimensions of the dataset: ", df.shape)
print("---------------------")

print("First 5 rows of the dataset: ")
print(df.head())

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
savedir = os.path.join(project_root, "data", "raw")
os.makedirs(savedir, exist_ok=True)

df.to_csv(os.path.join(savedir, "german_credit_data.csv"), index=False)

print(f"Data saved to {os.path.join(savedir, 'german_credit_data.csv')}")