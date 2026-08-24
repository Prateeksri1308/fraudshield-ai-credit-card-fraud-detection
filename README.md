# fraudshield-ai-credit-card-fraud-detection-
End-to-end ML system for credit card fraud detection using Scikit-learn, Random Forest, imbalanced-data evaluation, probability-based risk scoring, and Streamlit.
🛡️ FraudShield AI

Credit Card Fraud Detection & Transaction Risk Analysis

An end-to-end machine learning system for detecting fraudulent credit-card transaction patterns using Scikit-learn, Random Forest, imbalanced-data evaluation, and Streamlit.







🚨 Why FraudShield AI?

Credit-card fraud detection is not a normal binary-classification problem.

The dataset used in this project contains 284,807 transactions, of which only 492 are fraudulent — approximately 0.172% of all transactions.

That extreme class imbalance makes conventional accuracy a poor standalone indicator of model quality.

FraudShield AI focuses on the metrics that matter more for fraud detection:

Precision · Recall · F1 Score · ROC-AUC

The project follows an end-to-end workflow:

Raw Transaction Data
        ↓
Data Preparation
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
Class-Imbalance Experiments
        ↓
Multiple Model Evaluation
        ↓
Random Forest Selection
        ↓
Model Persistence
        ↓
Streamlit Application
        ↓
Transaction Risk Assessment

✨ Features

💳 Transaction Risk Analyzer

Analyze a real transaction from the dataset using the complete model feature vector.

The application keeps the technical PCA features away from the normal user interface while passing the required features to the trained model internally.

🎲 Transaction Explorer

Select a transaction or generate a random transaction from the dataset and run it through the trained classifier.

📊 Fraud Intelligence Dashboard

Explore:

Total transactions

Fraud transactions

Legitimate transactions

Fraud rate

Transaction amount statistics

Class distribution

🤖 Model Comparison

Evaluate multiple classification approaches before selecting the final model.

Models evaluated include:

Random Forest

Random Forest (Balanced)

SMOTE + Random Forest

KNN

Logistic Regression

Gradient Boosting

SMOTE + Logistic Regression

Logistic Regression (Balanced)

Naive Bayes

🔬 Technical Transparency

The application explains:

Model feature order

Time

Amount

V1–V28

Class labels

Model metrics

Dataset structure

🛡️ Risk Classification

The application converts model fraud probability into an easy-to-understand risk level:

Fraud Probability

Risk

< 25%

🟢 LOW

25–49%

🟡 MEDIUM

50–74%

🟠 HIGH

≥ 75%

🔴 CRITICAL

These application thresholds are presentation thresholds, not banking or regulatory standards.

🧠 Machine Learning

Dataset

FraudShield AI uses the well-known Credit Card Fraud Detection dataset containing transactions made by European cardholders in September 2013.

Dataset characteristics:

Property

Value

Total transactions

284,807

Fraudulent transactions

492

Legitimate transactions

284,315

Fraud rate

~0.172%

Input features

30

Target

Class

Feature structure

The dataset contains:

Time
V1
V2
V3
...
V28
Amount
Class

Class is the target:

0 → Legitimate
1 → Fraud

What are V1–V28?

V1 through V28 are PCA-transformed principal components.

The original transaction features were not provided because of confidentiality constraints.

Therefore, this project deliberately does not invent meanings such as:

V1 = income
V2 = merchant risk
V3 = location

Those interpretations would not be supported by the dataset.

Instead, FraudShield AI treats them correctly as anonymized numerical model features.

Time

Time represents the number of seconds elapsed between a transaction and the first transaction in the dataset.

It is not a clock timestamp.

Amount

Amount represents the transaction amount recorded in the dataset.

🏆 Model Performance

The following results were obtained during model evaluation:

Model

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Random Forest

99.95%

95.12%

79.59%

86.67%

91.52%

Random Forest (Balanced)

99.95%

97.44%

77.55%

86.36%

91.58%

SMOTE + Random Forest

99.94%

92.86%

79.59%

85.71%

97.66%

KNN

99.93%

86.67%

79.59%

82.98%

90.80%

Logistic Regression

99.85%

74.19%

46.94%

57.50%

92.30%

Gradient Boosting

99.85%

69.70%

46.94%

56.10%

63.21%

SMOTE + Logistic Regression

98.21%

8.71%

79.59%

15.69%

93.49%

Logistic Regression (Balanced)

98.18%

8.57%

79.59%

15.48%

93.73%

Naive Bayes

97.81%

7.21%

79.59%

13.22%

92.29%

Selected model

Random Forest

Why?

It provided a strong balance between:

Precision

Recall

F1 Score

ROC-AUC

The final choice should not be interpreted from accuracy alone.

An especially important observation from the experiments is that SMOTE + Random Forest achieved a higher ROC-AUC (97.66%), showing why model selection in an imbalanced fraud problem should consider the evaluation objective and decision threshold rather than simply selecting the highest accuracy.

📌 Why Accuracy Can Be Misleading

Imagine a dataset where almost every transaction is legitimate.

A model that predicts:

Every transaction → Legitimate

could achieve extremely high accuracy while detecting almost no fraud.

That is why FraudShield AI evaluates:

Precision

Of the transactions predicted as fraud, how many were actually fraud?

Recall

Of all actual fraudulent transactions, how many did the model detect?

F1 Score

Harmonic balance between precision and recall.

ROC-AUC

How well the model separates the two classes across classification thresholds.

AUPRC

For highly imbalanced fraud datasets, Area Under the Precision-Recall Curve is especially valuable.

The original dataset description specifically recommends AUPRC because ordinary accuracy can be misleading under severe class imbalance.

🏗️ Project Architecture

FraudShield AI
│
├── Data
│   └── Credit Card Transactions
│
├── Preprocessing
│   ├── Feature preparation
│   ├── Train/Test split
│   └── Class-imbalance experiments
│
├── Model Evaluation
│   ├── Random Forest
│   ├── Balanced Random Forest
│   ├── SMOTE + Random Forest
│   ├── KNN
│   ├── Logistic Regression
│   ├── Gradient Boosting
│   └── Naive Bayes
│
├── Final Model
│   └── Random Forest
│
├── Persistence
│   ├── fraud_model.pkl
│   └── model_metadata.pkl
│
└── Application
    └── Streamlit

📁 Repository Structure

fraudshield-ai-credit-card-fraud-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── model_metadata.pkl
│
└── assets/
    └── screenshots/
        ├── dashboard.png
        ├── assessment.png
        └── analytics.png

Keep large/private datasets and generated model artifacts out of GitHub when required by the dataset license or repository limits. The included .gitignore is intentionally conservative; adjust it according to what you intend to publish.

⚙️ Tech Stack

Technology

Purpose

Python

Core development

Pandas

Data manipulation

NumPy

Numerical processing

Scikit-learn

ML models, preprocessing & evaluation

Joblib

Model persistence

Matplotlib

Evaluation visualizations

Streamlit

Interactive ML application

Git & GitHub

Version control & collaboration

🚀 Run Locally

1. Clone the repository

git clone https://github.com/Prateeksri1308/fraudshield-ai-credit-card-fraud-detection.git
cd fraudshield-ai-credit-card-fraud-detection

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Verify the project structure

Make sure the application can access:

data/creditcard.csv
models/fraud_model.pkl

and, if used:

models/model_metadata.pkl

5. Start FraudShield AI

streamlit run app.py

The application will open in your browser.

☁️ Streamlit Deployment

The application is designed to run as a Streamlit web application.

Typical deployment configuration:

Main file:
app.py

Install dependencies from:

requirements.txt

Make sure required model/data artifacts are available to the deployed application and comply with the dataset's redistribution terms.

🔐 Important Design Decision

FraudShield AI does not ask normal users to manually enter:

V1
V2
V3
...
V28

Why?

Because these variables are anonymized PCA components.

Instead, the application uses a complete transaction row from the dataset:

Time + V1–V28 + Amount

and passes the correct feature vector to the trained model.

This prevents a misleading interface where users provide only Amount + Time while the model silently receives fabricated values for the remaining features.

⚠️ Limitations

This project is an educational machine-learning demonstration.

It should not be treated as:

A banking fraud prevention system

A production payment-security system

A financial decision engine

A real-time fraud investigation platform

A replacement for bank/card-provider fraud controls

The dataset is historical and contains anonymized PCA features.

Real-world fraud detection systems generally require additional information such as:

Merchant context

Device information

Account history

Geographic signals

Transaction velocity

Behavioral patterns

Real-time streaming data

Feedback from confirmed fraud cases

Those signals are not available in this dataset.

📚 Dataset & Research

The dataset was collected through a research collaboration involving Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles).

The dataset description recommends particular care when evaluating models because of its extreme class imbalance.

Relevant research includes work by:

Andrea Dal Pozzolo

Olivier Caelen

Reid A. Johnson

Gianluca Bontempi

Yann-Aël Le Borgne

Fabrizio Carcillo

Bertrand Lebichot

See the original dataset source and associated research for the complete attribution and citation requirements.

🧪 Future Improvements

Potential next steps include:

Precision-Recall curve visualization

AUPRC-based model selection

Probability calibration

Threshold optimization

Explainable AI with model-compatible feature attribution

Cross-validation

Time-aware validation

Streaming transaction simulation

Transaction velocity features

Real-time monitoring

Model drift detection

Experiment tracking

CI/CD automated testing

Containerized deployment

These are intentionally future improvements rather than claims about the current implementation.

🎯 Project Objective

FraudShield AI was built to demonstrate an end-to-end machine-learning workflow for a difficult real-world classification problem:

How can machine learning identify a tiny number of fraudulent transactions hidden inside a massive number of legitimate transactions?

The project combines:

Data Analysis
      +
Imbalanced Classification
      +
Model Benchmarking
      +
Random Forest
      +
Evaluation
      +
Model Persistence
      +
Streamlit
      =
FraudShield AI

👨‍💻 Author

Prateek Srivastava

Building projects around:

Machine Learning · Backend Development · AI/ML · Cloud · DevOps

⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore the implementation.

Disclaimer

FraudShield AI is an educational machine-learning and cybersecurity-awareness demonstration. It is not a banking, financial, fraud-investigation, or transaction-approval system. Model predictions are estimates and should not be used as the sole basis for real-world financial decisions.