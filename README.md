# 🛡️ FraudShield AI

> **Fast credit-card fraud risk assessment powered by machine learning.**

FraudShield AI is an end-to-end machine-learning project that analyzes credit-card transaction patterns and estimates the probability of fraudulent activity.

The project combines **data analysis, imbalanced-classification techniques, model comparison, Random Forest, and an interactive Streamlit application** into one practical fraud-detection system.

---

## 🚀 Project Overview

Credit-card fraud detection is a challenging machine-learning problem because fraudulent transactions are extremely rare compared with legitimate transactions.

FraudShield AI addresses this problem by:

- Analyzing a highly imbalanced fraud dataset
- Comparing multiple machine-learning models
- Evaluating models using fraud-focused metrics
- Selecting the best-performing model
- Saving the trained model for deployment
- Providing a fast Streamlit interface for transaction assessment

### Core Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing
     │
     ├── Median Imputation
     └── StandardScaler
     │
     ▼
Model Comparison
     │
     ├── Random Forest
     ├── Balanced Random Forest
     ├── SMOTE + Random Forest
     ├── KNN
     ├── Logistic Regression
     ├── Gradient Boosting
     └── Naive Bayes
     │
     ▼
Best Model Selection
     │
     ▼
Random Forest
     │
     ▼
Streamlit Application
     │
     ▼
Fraud Probability + Risk Classification
```

---

# ✨ Features

## 🔎 Quick Transaction Scan

Select a transaction using its ID and let FraudShield AI analyze it.

```text
Transaction ID
      ↓
Transaction Data
      ↓
Machine Learning Model
      ↓
Fraud Probability
      ↓
Risk Classification
```

---

## 🎲 Random Transaction Testing

Generate a random transaction instantly to experiment with the model.

Useful for:

- Demonstrations
- Testing
- Model exploration
- Portfolio presentations

---

## 🧠 Probability-Based Detection

Instead of simply returning:

```text
Fraud
```

or:

```text
Legitimate
```

FraudShield AI also provides the model's estimated probability:

```text
Fraud Probability: 0.03%
Legitimate Probability: 99.97%
```

This makes the result easier to understand.

---

## ⚡ Fast Inference

The application is optimized for a responsive Streamlit experience.

It uses:

- `st.cache_resource`
- `st.cache_data`
- Cached model loading
- Cached dataset loading
- Cached repeated predictions
- Lightweight transaction selection
- No massive transaction dropdown
- No model retraining during inference

---

## 📊 Dataset Insights

The application provides a simple dataset overview including:

- Total transactions
- Fraud transactions
- Legitimate transactions
- Fraud percentage
- Transaction amount statistics
- Class distribution

---

## 🤖 Model Performance

FraudShield AI displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- AUPRC when available

---

## 🔬 Technical Transparency

Advanced model inputs can be viewed through the technical section.

The model uses:

```text
Time
V1
V2
V3
...
V28
Amount
```

The target variable is:

```text
Class
```

where:

```text
0 = Legitimate
1 = Fraud
```

---

# 📊 Dataset

FraudShield AI uses the **Credit Card Fraud Detection** dataset.

The dataset contains transactions made by European cardholders during September 2013.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Transactions | 284,807 |
| Fraud Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Fraud Rate | ~0.172% |
| Input Features | 30 |
| Target Feature | `Class` |

The dataset is extremely imbalanced.

```text
Legitimate ≈ 99.828%
Fraud       ≈ 0.172%
```

---

# 🧬 Dataset Features

The dataset contains:

```text
Time
V1
V2
V3
...
V28
Amount
Class
```

---

## ⏱️ Time

`Time` represents the number of seconds elapsed between a transaction and the first transaction in the dataset.

It is **not a clock time**.

For example:

```text
120 seconds
≈ 2 minutes
≈ 0.03 hours
```

---

## 💰 Amount

`Amount` represents the transaction amount recorded in the dataset.

The original dataset does not provide a currency specification.

---

## 🧩 V1–V28

`V1` through `V28` are anonymized principal components obtained through PCA transformation.

Their original meanings are not publicly provided because of confidentiality.

Therefore, FraudShield AI does **not** assign unsupported meanings such as:

```text
V1 = income
V2 = location
V3 = merchant risk
```

Instead, they are treated strictly as numerical model features.

---

# 🧠 Machine Learning Pipeline

The final model uses a preprocessing and classification pipeline.

```text
Input Features
      │
      ▼
Median Imputation
      │
      ▼
StandardScaler
      │
      ▼
Random Forest
      │
      ▼
Prediction
      │
      ├── Class
      └── Probability
```

### Model Configuration

```text
Algorithm: Random Forest
Number of Trees: 150
```

The trained model is stored as:

```text
models/fraud_model.pkl
```

Metadata is stored as:

```text
models/model_metadata.pkl
```

---

# 🏆 Model Comparison

Multiple models were tested before selecting the final model.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **99.95%** | **95.12%** | **79.59%** | **86.67%** | **91.52%** |
| Random Forest (Balanced) | 99.95% | 97.44% | 77.55% | 86.36% | 91.58% |
| SMOTE + Random Forest | 99.94% | 92.86% | 79.59% | 85.71% | **97.66%** |
| KNN | 99.93% | 86.67% | 79.59% | 82.98% | 90.80% |
| Logistic Regression | 99.85% | 74.19% | 46.94% | 57.50% | 92.30% |
| Gradient Boosting | 99.85% | 69.70% | 46.94% | 56.10% | 63.21% |
| SMOTE + Logistic Regression | 98.21% | 8.71% | 79.59% | 15.69% | 93.49% |
| Logistic Regression (Balanced) | 98.18% | 8.57% | 79.59% | 15.48% | 93.73% |
| Naive Bayes | 97.81% | 7.21% | 79.59% | 13.22% | 92.29% |

---

# 🥇 Why Random Forest?

The final Random Forest provided a strong overall balance between:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Accuracy

The model achieved:

```text
Accuracy   → 99.95%
Precision  → 95.12%
Recall     → 79.59%
F1 Score   → 86.67%
ROC-AUC    → 91.52%
```

Although the SMOTE + Random Forest experiment achieved a higher ROC-AUC, the original Random Forest provided a strong overall balance and was selected as the final application model.

---

# ⚠️ Why Accuracy Is Not Enough

Fraud detection is a highly imbalanced classification problem.

Only approximately:

```text
0.172%
```

of transactions are fraudulent.

Because of this, a model can achieve extremely high accuracy while still failing to detect important fraudulent transactions.

Therefore, FraudShield AI focuses on:

### Precision

> Of the transactions predicted as fraud, how many were actually fraud?

### Recall

> Of the actual fraudulent transactions, how many were detected?

### F1 Score

> Harmonic balance between precision and recall.

### ROC-AUC

> How effectively the model separates legitimate and fraudulent transactions across classification thresholds.

### AUPRC

> Area Under the Precision-Recall Curve.

For highly imbalanced fraud detection, AUPRC is particularly useful.

---

# 🖥️ Application Interface

FraudShield AI contains three main sections:

```text
┌────────────────────────────────────────┐
│          🛡️ FraudShield AI             │
│                                        │
│       Scan   Insights   Model          │
└────────────────────────────────────────┘
```

---

## 1. 🔎 Scan

The primary feature.

```text
Transaction ID
       ↓
Transaction Details
       ↓
🔍 Scan Transaction
       ↓
ML Prediction
       ↓
Fraud Probability
       ↓
Risk Classification
```

Example:

```text
🟢 LOW FRAUD RISK

0.03%

Estimated probability of fraud

Fraud Probability        0.03%
Legitimate Probability   99.97%
Prediction               Legitimate
```

---

## 2. 📊 Insights

Provides a quick overview of the dataset.

```text
Total Transactions
Fraud Transactions
Fraud Rate
Average Amount
Class Distribution
```

---

## 3. 🤖 Model

Displays the selected model and evaluation metrics.

```text
Model
Accuracy
Precision
Recall
F1 Score
ROC-AUC
AUPRC
```

It also shows the model pipeline:

```text
Median Imputation
       ↓
StandardScaler
       ↓
Random Forest
       ↓
Fraud Probability
```

---

# 📁 Project Structure

```text
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
└── notebook/
    └── fraudshield_ai.ipynb
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Streamlit | Web application |
| Jupyter Notebook | Model experimentation |
| Git | Version control |
| GitHub | Repository & deployment |

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Prateeksri1308/fraudshield-ai-credit-card-fraud-detection.git
```

```bash
cd fraudshield-ai-credit-card-fraud-detection
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# ☁️ Deployment

FraudShield AI can be deployed using Streamlit Community Cloud.

### Main configuration

```text
Repository:
fraudshield-ai-credit-card-fraud-detection

Branch:
main

Main file:
app.py
```

Dependencies are installed from:

```text
requirements.txt
```

Make sure the trained model files are available:

```text
models/
├── fraud_model.pkl
└── model_metadata.pkl
```

---

# ⚡ Performance Architecture

The application was designed to avoid unnecessary computation during user interaction.

### Model caching

```python
@st.cache_resource
def load_model():
    ...
```

The model is loaded once and reused.

---

### Dataset caching

```python
@st.cache_data
def load_dataset():
    ...
```

The dataset is loaded once and reused.

---

### Prediction caching

Repeated scans of the same transaction can reuse the previous prediction.

```text
First Scan
   ↓
Model Inference
   ↓
Cache

Same Transaction
   ↓
Cached Result
   ↓
Instant Response
```

---

### Lightweight transaction selection

Instead of rendering hundreds of thousands of transaction choices:

```text
❌ Massive dropdown
```

FraudShield AI uses:

```text
Transaction ID
```

which keeps the interface simple and responsive.

---

# 🔐 Model Input Design

Users are intentionally **not required to enter V1–V28 manually**.

The application internally retrieves the complete transaction:

```text
Time
V1
V2
...
V28
Amount
```

and sends those features to the saved model.

The target:

```text
Class
```

is never sent to the model.

---

# 🧪 Example

Suppose a transaction has:

```text
Transaction ID: 15243
Amount:         360.00
Time:           120 seconds
```

FraudShield AI internally retrieves:

```text
Time
V1
V2
...
V28
Amount
```

Then:

```text
              ┌─────────────────┐
              │ Transaction     │
              │ Feature Vector  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Preprocessing   │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Random Forest   │
              └────────┬────────┘
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
       Prediction          Probability
             ↓                   ↓
       Legitimate              0.03%
```

---

# 🔮 Future Improvements

FraudShield AI can be extended with:

```text
Real-time transaction streams
        ↓
Transaction velocity features
        ↓
Merchant risk signals
        ↓
Device fingerprinting
        ↓
Behavioral analytics
        ↓
Explainable AI
        ↓
SHAP explanations
        ↓
Model monitoring
        ↓
Drift detection
        ↓
Production fraud API
```

Possible technical improvements:

- [ ] Precision-Recall curve
- [ ] AUPRC optimization
- [ ] Probability calibration
- [ ] Threshold optimization
- [ ] SHAP explanations
- [ ] Time-aware validation
- [ ] Cross-validation
- [ ] Model drift monitoring
- [ ] Real-time inference API
- [ ] Docker deployment
- [ ] CI/CD pipeline
- [ ] Experiment tracking
- [ ] Production database integration

---

# 📚 Dataset Attribution

The dataset was collected and analyzed during a research collaboration involving:

- Worldline
- Machine Learning Group
- Université Libre de Bruxelles (ULB)

The dataset and associated research should be cited according to the original dataset documentation.

Important related research includes work by:

- Andrea Dal Pozzolo
- Olivier Caelen
- Reid A. Johnson
- Gianluca Bontempi
- Yann-Aël Le Borgne
- Fabrizio Carcillo
- Bertrand Lebichot

---

# ⚠️ Limitations

This project uses a historical and anonymized dataset.

The real-world fraud detection environment can contain many additional signals that are unavailable in this dataset.

Examples include:

```text
Device
Merchant
Location
Account history
Transaction velocity
User behavior
IP information
Authentication signals
Real-time transaction history
```

These are **not represented** by the current dataset.

---

# ⚠️ Disclaimer

> **FraudShield AI is an educational machine-learning demonstration.**

It is not intended to function as:

- A banking fraud-detection system
- A payment-security platform
- A financial decision engine
- A fraud-investigation service
- A transaction approval system

Model predictions should **not** be used as the sole basis for real-world financial decisions.

---

# 👨‍💻 Author

## Prateek Srivastava

**AI/ML Engineer · Backend Developer · Cloud & DevOps**

Building practical software and machine-learning systems focused on real-world problems.

---

# ⭐ Why This Project Matters

FraudShield AI is not just a classification model.

It demonstrates the complete machine-learning workflow:

```text
Problem
  ↓
Dataset
  ↓
EDA
  ↓
Data Cleaning
  ↓
Imbalanced Classification
  ↓
Multiple Models
  ↓
Evaluation
  ↓
Model Selection
  ↓
Model Persistence
  ↓
Application Development
  ↓
Deployment
```

That makes the project a practical demonstration of:

**Machine Learning + Data Science + Model Evaluation + Deployment + Software Engineering**

---

<p align="center">

### 🛡️ FraudShield AI

**Detect patterns. Estimate risk. Build smarter systems.**

</p>