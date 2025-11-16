# Prédiction Conforme et Régression Quantile sur le Risque de Crédit

Projet d'analyse et de modélisation du risque de crédit utilisant des techniques avancées de machine learning avec quantification de l'incertitude : **Conformal Prediction** et **Régression Quantile**.

## Introduction

### Pourquoi le secteur financier et le risque de crédit ?

Le secteur financier est au cœur du fonctionnement de notre économie mondiale. Il facilite l'allocation du capital, finance l'innovation, soutient les entreprises et permet aux individus de réaliser leurs projets. L'évaluation du risque de crédit est l'une des fonctions les plus critiques de ce système : elle détermine qui peut emprunter, à quelles conditions, et avec quel niveau de risque pour le prêteur.

Ce projet s'inscrit dans ma passion pour la finance, un domaine qui non seulement m'anime intellectuellement, mais qui représente aussi un pilier fondamental de notre société. Comprendre et modéliser le risque de crédit, c'est contribuer à un système financier plus stable et plus équitable.

### L'importance de la quantification de l'incertitude

Dans le domaine financier, et particulièrement pour l'évaluation du risque de crédit, la **quantification de l'incertitude** n'est pas un luxe mais une nécessité réglementaire et opérationnelle :

- **Exigences réglementaires** : Les accords de Bâle imposent aux institutions financières de mesurer et de provisionner leurs risques avec des garanties statistiques
- **Gestion du risque** : Une prédiction ponctuelle ne suffit pas ; il faut des intervalles de confiance pour prendre des décisions éclairées
- **Confiance des investisseurs** : La transparence sur l'incertitude des modèles renforce la crédibilité des institutions
- **Stabilité financière** : Des modèles robustes avec quantification de l'incertitude contribuent à prévenir les crises systémiques

Les méthodes traditionnelles de machine learning fournissent des prédictions ponctuelles mais ne garantissent pas toujours une couverture fiable de l'incertitude. C'est là qu'interviennent **Conformal Prediction** et **Régression Quantile**, deux approches complémentaires qui offrent des garanties statistiques rigoureuses, essentielles dans un contexte aussi sensible que le crédit financier.

## Vue d'ensemble

Ce projet académique explore l'application de méthodes statistiques modernes pour l'évaluation du risque de crédit, avec un accent particulier sur la quantification de l'incertitude des prédictions. Deux approches complémentaires sont implémentées :

- **Régression Quantile** : Pour estimer les intervalles de prédiction des montants de prêt à différents niveaux de risque
- **Prédiction Conforme (Conformal Prediction)** : Pour la classification des ratings de crédit et la prédiction des montants de prêt avec garanties statistiques

## Objectifs

- Analyser et prétraiter des données financières de crédit
- Implémenter des modèles de prédiction avec quantification de l'incertitude
- Comparer Conformal Prediction et Régression Quantile sur des tâches de régression
- Développer une application professionnelle d'évaluation du risque de crédit
- Fournir des garanties statistiques sur les prédictions (90% de confiance)

## Structure du Projet

```
├── data/
│   ├── raw/                              # Données brutes depuis Kaggle
│   │   ├── corporate_rating.csv          # Dataset classification
│   │   └── Loan.csv                      # Dataset régression
│   └── processed/                        # Données prétraitées
├── src/
│   ├── data_loading.py                   # Téléchargement et chargement des données
│   ├── preprocessing.py                  # Prétraitement et feature engineering
│   └── Application_Professionnelle_Credit_Classification.py
│                                         # Application de production
├── notebooks/
│   ├── EDA_Classification.ipynb          # Exploration des données de classification
│   ├── EDA_Regression.ipynb              # Exploration des données de régression
│   ├── Prediction_Conforme_Classification.ipynb
│   │                                     # CP pour classification des ratings
│   ├── Prediction_Conforme_Regression.ipynb
│   │                                     # CP pour prédiction des montants
│   └── Regression_Quantile.ipynb         # Régression quantile comparative
└── reports/
    └── figures/                          # Visualisations et graphiques
```

## Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration Kaggle (optionnel)
Pour télécharger les datasets automatiquement depuis Kaggle :
```bash
# Placer votre kaggle.json dans ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Datasets

### Classification : Corporate Credit Rating
- **Source** : [Kaggle - Corporate Credit Rating](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating)
- **Objectif** : Prédire le rating de crédit d'une entreprise (AAA, AA, A, BBB, etc.)
- **Variables** : Ratios financiers (liquidité, rentabilité, solvabilité), données sectorielles

### Régression : Financial Risk for Loan Approval
- **Source** : [Kaggle - Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)
- **Objectif** : Prédire le risque d'un client qui souhaite contracter un prêt
- **Variables** : Revenus, historique de crédit, emploi, actifs, dettes

## Notebooks

### Analyse Exploratoire (EDA)
- `EDA_Classification.ipynb` : Analyse des données de rating de crédit
- `EDA_Regression.ipynb` : Analyse des données de prêt

### Modélisation
- `Prediction_Conforme_Classification.ipynb` : CP pour classification des ratings
- `Prediction_Conforme_Regression.ipynb` : CP pour prédiction des montants de prêt
- `Regression_Quantile.ipynb` : Mise en place de la régression quantile

## Auteur

**Lesech Erwann**

Projet académique réalisé dans le cadre du module de **Stochastiques** à EPITA.

## License

Voir le fichier [LICENSE](LICENSE) pour les détails.
