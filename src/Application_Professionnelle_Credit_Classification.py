"""
Application Professionnelle - Évaluation du Risque de Crédit

Système de prédiction conforme pour l'évaluation du risque de crédit.
Utilise la méthode Split Conformal Prediction (SCP) pour générer des 
ensembles de ratings avec garanties statistiques (90% de confiance).

Usage:
    Exécuter après avoir entraîné le modèle SCP dans 
    Prediction_Conforme_Classification.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CreditRatingPredictor:
    """Système de prédiction conforme pour l'évaluation du risque de crédit."""
    
    def __init__(self, scp_model, label_encoder, feature_encoders):
        """Initialise le prédicteur avec les modèles calibrés."""
        self.scp = scp_model
        self.le = label_encoder
        self.feature_encoders = feature_encoders
        
    def preprocess_input(self, company_data):
        """Prétraite les données et calcule les ratios financiers."""
        data = company_data.copy()
        total_assets = data.get('Total Assets', 0)
        total_liabilities = data.get('Total Liabilities', 0)
        total_equity = data.get('Total Equity', 0)
        total_revenue = data.get('Total Revenue', 0)
        net_income = data.get('Net Income', 0)
        ebitda = data.get('EBITDA', 0)
        market_cap = data.get('Market Capitalization', 0)
        
        # Calcul des ratios financiers
        current_assets = total_assets * 0.4
        current_liabilities = total_liabilities * 0.6
        cash = total_assets * 0.1
        
        ratios = {
            # Liquidité
            'currentRatio': current_assets / current_liabilities if current_liabilities > 0 else 1.5,
            'quickRatio': (current_assets - total_assets * 0.1) / current_liabilities if current_liabilities > 0 else 1.0,
            'cashRatio': cash / current_liabilities if current_liabilities > 0 else 0.5,
            'daysOfSalesOutstanding': 45.0,
            'assetTurnover': total_revenue / total_assets if total_assets > 0 else 0.8,
            'fixedAssetTurnover': total_revenue / (total_assets * 0.6) if total_assets > 0 else 1.2,
            'payablesTurnover': 8.0,
            'netProfitMargin': net_income / total_revenue if total_revenue > 0 else 0.1,
            'pretaxProfitMargin': (net_income * 1.25) / total_revenue if total_revenue > 0 else 0.12,
            'grossProfitMargin': 0.35,  # Valeur moyenne
            'operatingProfitMargin': ebitda / total_revenue if total_revenue > 0 else 0.15,
            'returnOnAssets': net_income / total_assets if total_assets > 0 else 0.08,
            'returnOnEquity': net_income / total_equity if total_equity > 0 else 0.15,
            'returnOnCapitalEmployed': ebitda / (total_assets - current_liabilities) if (total_assets - current_liabilities) > 0 else 0.12,
            'ebitPerRevenue': ebitda / total_revenue if total_revenue > 0 else 0.2,
            'debtEquityRatio': total_liabilities / total_equity if total_equity > 0 else 1.0,
            'debtRatio': total_liabilities / total_assets if total_assets > 0 else 0.5,
            'companyEquityMultiplier': total_assets / total_equity if total_equity > 0 else 2.0,
            'effectiveTaxRate': 0.25,
            'freeCashFlowOperatingCashFlowRatio': 0.7,
            'freeCashFlowPerShare': (ebitda * 0.6) / (market_cap / 100) if market_cap > 0 else 2.0,
            'cashPerShare': cash / (market_cap / 100) if market_cap > 0 else 5.0,
            'operatingCashFlowPerShare': ebitda / (market_cap / 100) if market_cap > 0 else 3.0,
            'operatingCashFlowSalesRatio': ebitda / total_revenue if total_revenue > 0 else 0.15,
            'enterpriseValueMultiple': market_cap / ebitda if ebitda > 0 else 10.0,
        }
        
        df = pd.DataFrame([ratios])
        if 'Date' in data:
            date_obj = pd.to_datetime(data['Date'])
            df['Year'] = date_obj.year
        else:
            df['Year'] = 2024
        
        df['Rating Agency Name'] = 0
        df['Sector'] = 0
        
        if 'Rating Agency Name' in data:
            agency = data['Rating Agency Name']
            if 'Rating Agency Name' in self.feature_encoders:
                encoder = self.feature_encoders['Rating Agency Name']
                if agency in encoder.classes_:
                    df['Rating Agency Name'] = encoder.transform([agency])[0]
        
        if 'Sector' in data:
            sector = data['Sector']
            if 'Sector' in self.feature_encoders:
                encoder = self.feature_encoders['Sector']
                if sector in encoder.classes_:
                    df['Sector'] = encoder.transform([sector])[0]
        
        expected_columns = [
            'Rating Agency Name', 'Sector', 'currentRatio', 'quickRatio', 'cashRatio',
            'daysOfSalesOutstanding', 'netProfitMargin', 'pretaxProfitMargin', 
            'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
            'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 
            'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio', 'effectiveTaxRate',
            'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
            'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
            'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover',
            'Year'
        ]
        
        df = df[expected_columns]
        
        return df
    
    def predict_with_interpretation(self, company_data):
        """Effectue une prédiction conforme avec interprétation."""
        X = self.preprocess_input(company_data)
        
        prediction_set = self.scp.predict(X)[0]
        pred_ratings = [self.le.classes_[c] for c in prediction_set]
        
        probs = self.scp.model.predict_proba(X)[0]
        classical_pred = self.le.classes_[np.argmax(probs)]
        classical_confidence = np.max(probs)
        set_size = len(prediction_set)
        
        if set_size == 1:
            interpretation = "✓ Forte confiance - Décision claire"
            recommendation = "Le modèle est très confiant. Vous pouvez procéder à l'évaluation."
        elif set_size == 2:
            interpretation = "⚠ Confiance modérée - Incertitude entre 2 classes"
            recommendation = "Analyse complémentaire recommandée pour départager les 2 ratings possibles."
        else:
            interpretation = "⚠⚠ Faible confiance - Forte incertitude"
            recommendation = "Incertitude élevée. Une analyse approfondie est nécessaire avant toute décision."
        
        return {
            'company_name': company_data.get('Name', 'Entreprise'),
            'conformal_set': pred_ratings,
            'set_size': set_size,
            'classical_prediction': classical_pred,
            'classical_confidence': classical_confidence,
            'interpretation': interpretation,
            'recommendation': recommendation
        }
    
    def display_prediction(self, result):
        """Affiche les résultats de la prédiction."""
        print("=" * 70)
        print(f"ÉVALUATION DU RISQUE DE CRÉDIT - {result['company_name']}")
        print("=" * 70)
        print()
        print("Prédiction Conforme (90% de confiance):")
        print(f"  Ratings possibles: {', '.join(result['conformal_set'])}")
        print(f"  Taille de l'ensemble: {result['set_size']}")
        print()
        print("Prédiction Classique:")
        print(f"  Rating prédit: {result['classical_prediction']}")
        print(f"  Confiance: {result['classical_confidence']:.1%}")
        print()
        print(f"Interprétation: {result['interpretation']}")
        print(f"  {result['recommendation']}")
        print()
        print("=" * 70)


def interactive_credit_evaluation(predictor):
    """Interface interactive pour évaluer une entreprise."""
    print("=" * 70)
    print("SYSTÈME D'ÉVALUATION DU RISQUE DE CRÉDIT")
    print("Prédiction Conforme avec 90% de confiance")
    print("=" * 70)
    print()
    
    company_data = {}
    
    print("Informations de l'entreprise:")
    company_data['Name'] = input("  Nom: ") or "Entreprise X"
    company_data['Symbol'] = input("  Symbole boursier: ") or "XXX"
    company_data['Date'] = input("  Date (YYYY-MM-DD): ") or "2024-06-15"
    
    print("\nInformations sectorielles:")
    print("  Agences: Standard & Poor's, Moody's, Fitch Ratings")
    company_data['Rating Agency Name'] = input("  Agence de notation: ") or "Standard & Poor's"
    
    print("  Secteurs: Technology, Financial Services, Healthcare, Consumer Goods, etc.")
    company_data['Sector'] = input("  Secteur: ") or "Technology"
    
    print("\nMétriques financières:")
    try:
        company_data['Total Assets'] = float(input("  Total des actifs: ") or 5000000)
        company_data['Total Liabilities'] = float(input("  Total des passifs: ") or 2000000)
        company_data['Total Revenue'] = float(input("  Chiffre d'affaires: ") or 3000000)
        company_data['Net Income'] = float(input("  Résultat net: ") or 450000)
        company_data['EBITDA'] = float(input("  EBITDA: ") or 600000)
        company_data['Total Equity'] = float(input("  Capitaux propres: ") or 3000000)
        company_data['Market Capitalization'] = float(input("  Capitalisation boursière: ") or 8000000)
    except ValueError:
        print("\nErreur de saisie. Utilisation des valeurs par défaut.")
        company_data.update({
            'Total Assets': 5000000,
            'Total Liabilities': 2000000,
            'Total Revenue': 3000000,
            'Net Income': 450000,
            'EBITDA': 600000,
            'Total Equity': 3000000,
            'Market Capitalization': 8000000
        })
    
    print("\nAnalyse en cours...")
    print()
    
    result = predictor.predict_with_interpretation(company_data)
    predictor.display_prediction(result)
    
    return result


def batch_credit_evaluation(predictor, companies_list):
    """Évalue plusieurs entreprises et retourne un rapport synthétique."""
    results = []
    
    for company in companies_list:
        result = predictor.predict_with_interpretation(company)
        results.append({
            'Entreprise': result['company_name'],
            'Ratings Possibles': ', '.join(result['conformal_set']),
            'Nombre de Ratings': result['set_size'],
            'Prédiction Classique': result['classical_prediction'],
            'Confiance Classique': f"{result['classical_confidence']:.1%}",
            'Statut': result['interpretation']
        })
    
    return pd.DataFrame(results)


def exemple_entreprise_saine():
    """Exemple d'entreprise avec de bonnes métriques financières."""
    return {
        'Name': 'TechCorp Solutions',
        'Symbol': 'TECH',
        'Date': '2024-06-15',
        'Rating Agency Name': "Standard & Poor's",
        'Sector': 'Technology',
        'Total Assets': 5000000,
        'Total Liabilities': 2000000,
        'Total Revenue': 3000000,
        'Net Income': 450000,
        'EBITDA': 600000,
        'Total Equity': 3000000,
        'Market Capitalization': 8000000
    }


def exemple_entreprise_risquee():
    """Exemple d'entreprise avec endettement élevé."""
    return {
        'Name': 'Retail Inc.',
        'Symbol': 'RETL',
        'Date': '2024-06-15',
        'Rating Agency Name': "Moody's",
        'Sector': 'Consumer Goods',
        'Total Assets': 2000000,
        'Total Liabilities': 1800000,
        'Total Revenue': 1500000,
        'Net Income': 50000,
        'EBITDA': 120000,
        'Total Equity': 200000,
        'Market Capitalization': 500000
    }


def exemple_batch():
    """Exemple d'analyse par lot de plusieurs entreprises."""
    return [
        {
            'Name': 'Alpha Corp',
            'Date': '2024-06-15',
            'Rating Agency Name': "Standard & Poor's",
            'Sector': 'Technology',
            'Total Assets': 10000000,
            'Total Liabilities': 3000000,
            'Total Revenue': 8000000,
            'Net Income': 1200000,
            'EBITDA': 1500000,
            'Total Equity': 7000000,
            'Market Capitalization': 15000000
        },
        {
            'Name': 'Beta Industries',
            'Date': '2024-06-15',
            'Rating Agency Name': "Moody's",
            'Sector': 'Manufacturing',
            'Total Assets': 5000000,
            'Total Liabilities': 4000000,
            'Total Revenue': 3000000,
            'Net Income': 200000,
            'EBITDA': 350000,
            'Total Equity': 1000000,
            'Market Capitalization': 2000000
        },
        {
            'Name': 'Gamma Services',
            'Date': '2024-06-15',
            'Rating Agency Name': 'Fitch Ratings',
            'Sector': 'Financial Services',
            'Total Assets': 3000000,
            'Total Liabilities': 2500000,
            'Total Revenue': 2000000,
            'Net Income': 100000,
            'EBITDA': 180000,
            'Total Equity': 500000,
            'Market Capitalization': 1200000
        }
    ]


if __name__ == "__main__":
    print("""
Application Professionnelle - Évaluation du Risque de Crédit
Prédiction Conforme avec garanties statistiques (90% confiance)

Pré-requis:
  Exécuter le notebook 'Prediction_Conforme_Classification.ipynb'
  pour entraîner et calibrer le modèle SCP.

Utilisation:

1. Importer et initialiser:
   >>> from Application_Professionnelle_Credit import CreditRatingPredictor
   >>> predictor = CreditRatingPredictor(scp, le, feature_encoders)

2. Évaluer une entreprise:
   >>> company = exemple_entreprise_saine()
   >>> result = predictor.predict_with_interpretation(company)
   >>> predictor.display_prediction(result)

3. Évaluation interactive:
   >>> interactive_credit_evaluation(predictor)

4. Analyse par lot:
   >>> companies = exemple_batch()
   >>> batch_results = batch_credit_evaluation(predictor, companies)
   >>> print(batch_results)
    """)
