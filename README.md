# Prédiction Conforme et Régression Quantile sur le Risque de Crédit

## Contexte
Ce projet s’inscrit dans le cadre d’une étude sur la **quantification de l’incertitude** dans les modèles de Machine Learning.  
L’objectif est d’analyser et de comparer plusieurs méthodes de construction d’intervalles ou d’ensembles de prédiction :
- **Régression quantile**  
- **Prédiction conforme pour la régression** (Jackknife+, CV+, Split)  
- **Prédiction conforme pour la classification** (SCP, FCP)

Ces méthodes sont appliquées au jeu de données **Statlog (German Credit Data)** afin d’évaluer leur performance et leur robustesse dans un contexte de **gestion du risque bancaire**.

---

## Objectifs du projet
1. Créer des **intervalles de prédiction** via la régression quantile.  
2. Créer des **intervalles conformes** via les méthodes Jackknife+ et CV+.  
3. Créer des **ensembles de prédiction conformes** pour la classification du risque de crédit.  
4. Analyser la couverture, la largeur moyenne des intervalles et la stabilité statistique.

---

## Jeu de données
- **Nom :** Statlog (German Credit Data)  
- **Source :** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
- **Taille :** 1000 instances, 20 attributs  
- **Cible :** `Good` / `Bad` (risque de crédit)

Chaque observation décrit un client bancaire (âge, statut, durée du crédit, montant, emploi, etc.) et indique s’il a bien remboursé son prêt.  
Le dataset est non temporel, ce qui permet d’appliquer les méthodes conformes sans biais séquentiel.

---

## Structure du projet

```sh
repository
│
├── data/
│   ├── raw/                # Données brutes téléchargées
│   └── processed/          # Données prêtes pour modélisation
│
├── notebooks/              # Notebooks exploratoires ou démonstratifs
|   ├── eda_overview.ipynb  # Exploration descriptive initiale
│
├── src/
│   ├── download_data.py    # Téléchargement + sauvegarde des données
│   ├── __init__.py
│
├── reports/
│   ├── figures/            # Graphiques de l’EDA
│   └── summary.md          # Notes d’analyse
│
├── requirements.txt        # Librairies Python nécessaires
├── README.md               # Présentation complète du projet
└── .gitignore
```

---

## Environnement Python

```bash
pip install -r requirements.txt
```

## Auteurs

- Lesech Erwann

---

Projet académique réalisé dans le cadre d’un module de Processus Stochastiques et Incertitude en Machine Learning, avec une approche scientifique et professionnelle.