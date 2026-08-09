import os
from dotenv import load_dotenv
from google import genai

# Automatically load variables from .env file
load_dotenv()

def generate_fraud_triage_report(transaction_summary: dict, top_features: list, api_key: str = None) -> str:
    # Uses key from sidebar UI, or falls back to .env / system environment
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if not effective_api_key:
        return (
            f"🚨 MOCK LLM TRIAGE REPORT: Transaction flagged due to critical anomalies in {top_features}. "
            f"Transaction amount of ${transaction_summary.get('Amount', 0):.2f} deviates significantly from expected pattern. "
            f"Action: Hold funds and notify cardholder."
        )
    
    try:
        client = genai.Client(api_key=effective_api_key)
        prompt = f"""
        You are an expert fraud investigator. An XGBoost anomaly engine flagged a credit card transaction.
        Transaction Summary: {transaction_summary}
        Key Anomalous Features identified by SHAP: {top_features}.
        Write a 2-sentence professional investigation report explaining why this transaction was flagged.
        """
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"