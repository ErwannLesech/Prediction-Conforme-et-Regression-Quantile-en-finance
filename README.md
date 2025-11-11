# Analyse de Données et EDA pour le Risque de Crédit

Ce projet fournit des outils pour télécharger, explorer et préparer des datasets de régression et classification dans le domaine du risque de crédit.

## 🎯 Objectifs

- Téléchargement automatique de datasets depuis Kaggle
- Analyse exploratoire des données (EDA) complète
- Préprocessing et nettoyage des données
- Préparation des données pour l'analyse

## 📁 Structure du Projet

```
├── data/
│   ├── raw/                    # Données brutes téléchargées
│   └── processed/              # Données nettoyées et préparées
├── src/
│   ├── data_loading.py         # Téléchargement et chargement des données
│   └── preprocessing.py        # Préparation et nettoyage des données
├── notebooks/
│   ├── 01_EDA_Regression.ipynb       # Analyse exploratoire régression
│   └── 02_EDA_Classification.ipynb   # Analyse exploratoire classification
├── reports/                    # Résultats et visualisations
└── eda_overview.ipynb         # Vue d'ensemble EDA originale
```

## 🚀 Installation et Utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

## 📊 Datasets

Le projet utilise des datasets provenant de Kaggle via `kagglehub`:

- **Régression**: Financial Risk for Loan Approval
  - Source: https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval
  - Variables: Données financières pour évaluation du risque de prêt
  
- **Classification**: Corporate Credit Rating  
  - Source: https://www.kaggle.com/datasets/agewerc/corporate-credit-rating
  - Variables: Données d'entreprises pour classification du rating de crédit

## 👨‍💻 Auteur

- Lesech Erwann

Projet académique réalisé dans le cadre d'un module de Processus Stochastiques et Incertitude en Machine Learning.

## 📄 License

Voir le fichier [LICENSE](LICENSE) pour les détails.