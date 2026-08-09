# 💳 Real-Time Credit Card Fraud Detection & GenAI Copilot

## 📌 Project Overview
This project is an end-to-end, production-ready machine learning pipeline designed to detect fraudulent credit card transactions in real-time. Moving beyond basic tabular classification, it features a modular architecture, an Explainable AI (XAI) layer using SHAP, and a Large Language Model (LLM) Copilot that automatically generates natural language triage reports for fraud analysts.

**Dataset Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## 🚀 Key Features
* **Extreme Class Imbalance Handling:** Utilizes SMOTE to balance a dataset where fraud represents less than 0.2% of transactions, preventing model bias.
* **High-Performance ML Engine:** Implements an XGBoost classifier optimized for Recall to minimize the financial loss of false negatives.
* **Automated MLOps:** Integrates MLflow to automatically track model hyperparameters, precision/recall metrics, and training artifacts via a local SQLite database.
* **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to mathematically extract the exact principal components (e.g., V14, V10) driving a fraud alert.
* **GenAI Fraud Copilot:** Leverages the Google Gemini 1.5 API to translate complex SHAP anomaly scores into instant, professional 2-sentence investigation reports.
* **Streaming Dashboard:** Features an interactive Streamlit frontend simulating a live payment gateway with millisecond inference latency and manual stress-testing capabilities.

## 🏗️ Architecture & Project Structure
* `data/`: Contains the raw Kaggle dataset (git-ignored).
* `models/`: Stores the serialized `xgboost_model.json` artifact for rapid inference.
* `src/train.py`: The data engineering and training script (SMOTE, XGBoost, MLflow).
* `src/inference.py`: The high-speed evaluation engine and SHAP explainer.
* `src/copilot.py`: Manages prompt engineering and the Gemini API connection.
* `app.py`: The Streamlit dashboard acting as the presentation and simulation layer.
* `mlflow.db`: Local tracking database for MLOps logging.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/fraud-detection-genai-copilot.git](https://github.com/yourusername/fraud-detection-genai-copilot.git)
   cd fraud-detection-genai-copilot

How to run!!!

1.Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

2.Install dependencies:

Bash
pip install -r requirements.txt

3.Configure Environment Variables:

Create a .env file in the root directory and add your free Gemini API key:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here

4: Download the Dataset
Go to the Kaggle Credit Card Fraud Dataset.

Download the ZIP file and extract it.

Rename the extracted file to creditcard.csv.

Create a folder named data inside the main project directory and place the CSV file there. The final path must be: data/creditcard.csv.


5: Train the Model
You only need to do this once. This script will read the CSV, balance the data with SMOTE, train the XGBoost model, log metrics to MLflow, and save the final model artifact.

Bash
python src/train.py
(Wait for the terminal to print "✅ Model saved successfully")

6: Launch the Application
Start the Streamlit web dashboard to simulate a live transaction stream and interact with the GenAI Copilot.

Bash
streamlit run app.py
The app will automatically open in your default web browser at http://localhost:8501.

👤 Author
Piyush Kuldeep
