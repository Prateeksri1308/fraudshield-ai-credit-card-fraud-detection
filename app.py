from __future__ import annotations

import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_CANDIDATES = [
    MODEL_DIR / "fraud_model.pkl",
    MODEL_DIR / "model.pkl",
    BASE_DIR / "fraud_model.pkl",
    BASE_DIR / "model.pkl",
]

METADATA_CANDIDATES = [
    MODEL_DIR / "model_metadata.pkl",
    MODEL_DIR / "metadata.pkl",
    BASE_DIR / "model_metadata.pkl",
]

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 42px;
        border-radius: 26px;
        background: linear-gradient(135deg,#0f172a,#172554);
        color: white;
        margin-bottom: 28px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(34,197,94,.12);
        border: 1px solid rgba(134,239,172,.3);
        color: #86efac;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .08em;
    }

    .hero h1 {
        font-size: 46px;
        line-height: 1.05;
        margin: 18px 0 10px;
        letter-spacing: -2px;
    }

    .hero p {
        max-width: 760px;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(15,23,42,.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 850;
        margin-top: 5px;
    }

    .risk {
        border-radius: 22px;
        padding: 28px;
        margin: 20px 0;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .risk.low { border-left: 6px solid #16a34a; }
    .risk.medium { border-left: 6px solid #d97706; }
    .risk.high { border-left: 6px solid #ea580c; }
    .risk.critical { border-left: 6px solid #dc2626; }

    .risk-score {
        font-size: 48px;
        font-weight: 900;
        color: #0f172a;
    }

    .signal {
        padding: 11px 13px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin: 7px 0;
        background: white;
        color: #334155;
        font-size: 13px;
    }

    .section-title {
        color: #0f172a;
        font-size: 25px;
        font-weight: 850;
        margin: 28px 0 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 18px;
        line-height: 1.65;
    }

    .footer {
        color: #94a3b8;
        font-size: 12px;
        text-align: center;
        line-height: 1.7;
        padding: 28px 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL / METADATA / DATA
# ============================================================

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path
    return None, None


@st.cache_resource
def load_metadata():
    for path in METADATA_CANDIDATES:
        if path.exists():
            return joblib.load(path), path
    return {}, None


@st.cache_data
def load_dataset():
    if not DATA_DIR.exists():
        return None, None

    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        return None, None

    preferred = [
        p for p in csvs
        if p.name.lower() in {
            "creditcard.csv",
            "credit_card.csv",
            "credit-card.csv",
        }
    ]

    path = preferred[0] if preferred else csvs[0]
    return pd.read_csv(path), path


model, model_path = load_model()
metadata, metadata_path = load_metadata()
dataset, dataset_path = load_dataset()


# ============================================================
# HELPERS
# ============================================================

def metadata_value(*keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def get_model_features():
    saved = metadata_value("features", "feature_names", "model_features")
    if saved:
        return list(saved)

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if dataset is not None:
        return [
            c for c in dataset.columns
            if c not in {"Class", "class", "Target", "target"}
        ]

    return []


def get_metrics():
    raw = metadata_value("metrics", default={}) or {}

    def value(*names):
        for name in names:
            if name in raw:
                try:
                    v = float(raw[name])
                    return v / 100 if v > 1 else v
                except (TypeError, ValueError):
                    pass
        return None

    return {
        "accuracy": value("accuracy", "Accuracy"),
        "precision": value("precision", "Precision"),
        "recall": value("recall", "Recall"),
        "f1": value("f1", "f1_score", "F1", "F1 Score"),
        "roc_auc": value("roc_auc", "ROC-AUC", "roc_auc_score"),
        "auprc": value(
            "auprc",
            "average_precision",
            "average_precision_score",
            "AUPRC",
        ),
    }


metrics = get_metrics()
MODEL_NAME = metadata_value("model", "model_name", default="Random Forest")
MODEL_FEATURES = get_model_features()


def format_metric(value):
    return "—" if value is None else f"{value:.2%}"


def classify_probability(probability):
    if probability >= 0.75:
        return "CRITICAL", "critical", "🔴"
    if probability >= 0.50:
        return "HIGH", "high", "🟠"
    if probability >= 0.25:
        return "MEDIUM", "medium", "🟡"
    return "LOW", "low", "🟢"


def find_class_index(classes, target):
    for i, value in enumerate(classes):
        try:
            if int(value) == target:
                return i
        except (TypeError, ValueError):
            if str(value) == str(target):
                return i
    return None


def make_model_input(row):
    if model is None:
        raise RuntimeError("Trained model could not be loaded.")

    if not MODEL_FEATURES:
        raise RuntimeError("Model feature order could not be determined.")

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature != "Hour" and feature not in row.index
    ]

    if missing:
        raise ValueError(
            "The selected transaction is missing required model features: "
            + ", ".join(missing)
        )

    values = {}

    for feature in MODEL_FEATURES:
        if feature == "Hour":
            if "Time" not in row.index:
                raise ValueError("Time is required to derive Hour.")
            values[feature] = int(float(row["Time"]) // 3600)
        else:
            values[feature] = row[feature]

    return pd.DataFrame(
        [[values[f] for f in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )


def predict_row(row):
    X = make_model_input(row)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    fraud_index = find_class_index(classes, 1)
    legitimate_index = find_class_index(classes, 0)

    fraud_probability = (
        float(probabilities[fraud_index])
        if fraud_index is not None
        else 0.0
    )

    legitimate_probability = (
        float(probabilities[legitimate_index])
        if legitimate_index is not None
        else 1.0 - fraud_probability
    )

    return (
        X,
        prediction,
        legitimate_probability,
        fraud_probability,
    )


def render_metric_cards(items):
    cols = st.columns(len(items))

    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_risk(fraud_probability, prediction):
    level, css, icon = classify_probability(fraud_probability)

    if prediction == 1:
        headline = "Fraudulent Pattern Detected"
        explanation = (
            "The model classified this transaction as fraud "
            "based on its learned transaction pattern."
        )
    else:
        headline = "Likely Legitimate"
        explanation = (
            "The model classified this transaction as legitimate "
            "based on its learned transaction pattern."
        )

    st.markdown(
        f"""
        <div class="risk {css}">
            <div style="font-size:28px;font-weight:900;color:#0f172a;">
                {icon} {level} FRAUD RISK
            </div>
            <div class="risk-score">
                {fraud_probability:.2%}
            </div>
            <strong>Estimated probability of fraud</strong>
            <p style="color:#64748b;margin-top:12px;">
                {headline}. {explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if level in {"HIGH", "CRITICAL"}:
        st.warning(
            "This is an ML estimate, not proof of fraud. "
            "For a real transaction, verify it through your "
            "bank or card provider's official channel."
        )
    else:
        st.info(
            "A low model score does not guarantee that a transaction "
            "is safe. This application is an educational demonstration."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.caption("Credit-card transaction fraud detection")

    page = st.radio(
        "Navigate",
        [
            "Transaction Analyzer",
            "Dataset Insights",
            "Model Comparison",
            "Technical Details",
        ],
    )

    st.divider()

    st.caption(
        "This app uses the trained model and the original "
        "transaction feature structure. V1–V28 are kept "
        "technical because their meanings are anonymized."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">CREDIT CARD FRAUD DETECTION</div>
        <h1>FraudShield AI</h1>
        <p>
            Detect suspicious credit-card transaction patterns
            using machine learning and a highly imbalanced
            real-world fraud dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRANSACTION ANALYZER
# ============================================================

if page == "Transaction Analyzer":

    st.markdown(
        '<div class="section-title">Transaction Analyzer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Select a real transaction from the dataset. FraudShield
            sends the complete trained feature vector to the model
            while keeping the anonymized PCA features out of the
            normal user interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "Trained model not found. Expected fraud_model.pkl "
            "inside the models/ folder."
        )

        if model_path:
            st.caption(f"Loaded from: {model_path}")

    elif dataset is None:
        st.error(
            "Transaction dataset not found. Put your CSV inside "
            "the data/ folder."
        )

    else:
        required = {"Time", "Amount", "Class"}

        if not required.issubset(dataset.columns):
            st.error(
                "The dataset must contain Time, Amount and Class."
            )
        else:

            working_df = dataset.dropna(
                subset=["Time", "Amount", "Class"]
            ).reset_index(drop=True)

            if working_df.empty:
                st.warning("No usable transactions were found.")
            else:

                if "selected_transaction" not in st.session_state:
                    st.session_state.selected_transaction = random.randrange(
                        len(working_df)
                    )

                c1, c2 = st.columns([3, 1])

                with c1:
                    index = st.selectbox(
                        "Transaction",
                        range(len(working_df)),
                        index=st.session_state.selected_transaction,
                        format_func=lambda i: (
                            f"Transaction #{i + 1:,}  ·  "
                            f"Amount {float(working_df.iloc[i]['Amount']):,.2f}"
                        ),
                    )
                    st.session_state.selected_transaction = index

                with c2:
                    st.write("")
                    st.write("")
                    if st.button(
                        "🎲 Random",
                        use_container_width=True,
                    ):
                        st.session_state.selected_transaction = random.randrange(
                            len(working_df)
                        )
                        st.rerun()

                row = working_df.iloc[index]

                elapsed_seconds = float(row["Time"])
                amount = float(row["Amount"])
                derived_hour = int(elapsed_seconds // 3600)

                render_metric_cards(
                    [
                        ("Transaction Amount", f"{amount:,.2f}"),
                        ("Elapsed Time", f"{elapsed_seconds:,.0f} sec"),
                        ("Derived Hour", str(derived_hour)),
                    ]
                )

                st.markdown(
                    f"""
                    <div class="card" style="margin-top:18px;">
                        <strong>About transaction time</strong>
                        <p style="color:#64748b;margin:8px 0 0;">
                            The dataset's Time value is the number of seconds
                            elapsed since the first transaction in the dataset.
                            It is not a real clock timestamp.
                            {elapsed_seconds:,.0f} seconds ≈
                            {elapsed_seconds / 60:,.1f} minutes.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                if st.button(
                    "🔍 Analyze Transaction",
                    type="primary",
                    use_container_width=True,
                ):

                    try:
                        (
                            model_input,
                            prediction,
                            legitimate_probability,
                            fraud_probability,
                        ) = predict_row(row)

                        st.session_state.last_prediction = {
                            "prediction": prediction,
                            "legitimate": legitimate_probability,
                            "fraud": fraud_probability,
                            "amount": amount,
                            "time": elapsed_seconds,
                            "hour": derived_hour,
                        }

                    except Exception as exc:
                        st.error("The transaction could not be analyzed.")

                        with st.expander("Technical details"):
                            st.code(str(exc))

                result = st.session_state.get("last_prediction")

                if result:

                    st.markdown(
                        '<div class="section-title">Assessment Result</div>',
                        unsafe_allow_html=True,
                    )

                    render_risk(
                        result["fraud"],
                        result["prediction"],
                    )

                    render_metric_cards(
                        [
                            (
                                "Fraud Probability",
                                f"{result['fraud']:.2%}",
                            ),
                            (
                                "Legitimate Probability",
                                f"{result['legitimate']:.2%}",
                            ),
                            (
                                "Prediction",
                                "Fraud"
                                if result["prediction"] == 1
                                else "Legitimate",
                            ),
                        ]
                    )

                    st.markdown("### Model assessment")

                    st.progress(
                        result["fraud"],
                        text=(
                            f"Fraud probability: "
                            f"{result['fraud']:.2%}"
                        ),
                    )

                    st.markdown(
                        """
                        <div class="card">
                            <strong>How to interpret this result</strong>
                            <p style="color:#64748b;">
                                The probability is produced by the trained
                                classifier from the complete transaction
                                feature vector. It is not a guarantee of
                                whether the transaction is actually fraudulent.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("🔬 Technical model input"):
                        st.caption(
                            "V1–V28 are anonymized PCA-transformed features. "
                            "Their individual meanings are not provided by "
                            "the source dataset."
                        )

                        st.dataframe(
                            model_input.T.rename(
                                columns={0: "Value"}
                            ),
                            use_container_width=True,
                        )


# ============================================================
# DATASET INSIGHTS
# ============================================================

elif page == "Dataset Insights":

    st.markdown(
        '<div class="section-title">Dataset Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            The dataset contains credit-card transactions from
            European cardholders over two days. The positive class
            is extremely rare, making fraud detection an imbalanced
            classification problem.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataset is None:
        st.error("Dataset not found in the data/ folder.")
    else:
        total = len(dataset)
        fraud_count = int((dataset["Class"] == 1).sum())
        legitimate_count = total - fraud_count
        fraud_rate = fraud_count / total if total else 0

        render_metric_cards(
            [
                ("Total Transactions", f"{total:,}"),
                ("Fraud Transactions", f"{fraud_count:,}"),
                ("Legitimate", f"{legitimate_count:,}"),
                ("Fraud Rate", f"{fraud_rate:.3%}"),
            ]
        )

        st.markdown("### Class distribution")

        distribution = pd.DataFrame(
            {
                "Class": ["Legitimate", "Fraud"],
                "Transactions": [
                    legitimate_count,
                    fraud_count,
                ],
            }
        )

        st.bar_chart(
            distribution.set_index("Class")
        )

        st.markdown("### Transaction amounts")

        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Legitimate transactions**")
            st.metric(
                "Average amount",
                f"{dataset.loc[dataset['Class'] == 0, 'Amount'].mean():,.2f}",
            )

        with a2:
            st.markdown("**Fraudulent transactions**")
            st.metric(
                "Average amount",
                f"{dataset.loc[dataset['Class'] == 1, 'Amount'].mean():,.2f}",
            )

        st.markdown(
            """
            <div class="card" style="margin-top:20px;">
                <strong>Why accuracy is not enough</strong>
                <p style="color:#64748b;">
                    With fraud representing only a tiny fraction of all
                    transactions, a model can achieve very high accuracy
                    while still missing important fraud cases. Precision,
                    recall, F1, ROC-AUC and especially AUPRC provide more
                    useful views of performance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Comparison from the model-evaluation stage of the project.
            These results should be interpreted in the context of the
            highly imbalanced fraud class.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        [
            ["Random Forest", 0.999487, 0.951220, 0.795918, 0.866667, 0.915169],
            ["Random Forest (Balanced)", 0.999487, 0.974359, 0.775510, 0.863636, 0.915815],
            ["SMOTE + Random Forest", 0.999444, 0.928571, 0.795918, 0.857143, 0.976569],
            ["KNN", 0.999316, 0.866667, 0.795918, 0.829787, 0.907974],
            ["Logistic Regression", 0.998546, 0.741935, 0.469388, 0.575000, 0.923049],
            ["Gradient Boosting", 0.998461, 0.696970, 0.469388, 0.560976, 0.632098],
            ["SMOTE + Logistic Regression", 0.982086, 0.087054, 0.795918, 0.156942, 0.934924],
            ["Logistic Regression (Balanced)", 0.981786, 0.085714, 0.795918, 0.154762, 0.937279],
            ["Naive Bayes", 0.978109, 0.072089, 0.795918, 0.132203, 0.922896],
        ],
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ],
    )

    display_df = comparison.copy()

    for column in display_df.columns[1:]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.2%}"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Why Random Forest?")

    st.markdown(
        """
        The selected Random Forest provides a strong balance between
        precision, recall and F1 in the evaluated experiments. The
        SMOTE + Random Forest experiment has a higher ROC-AUC in the
        supplied comparison, so the choice of final model should be
        justified by the project's selected evaluation criteria and
        thresholding strategy rather than accuracy alone.
        """
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

elif page == "Technical Details":

    st.markdown(
        '<div class="section-title">Technical Details</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Technical information about the dataset, model and
            feature representation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_cards(
        [
            ("Model", MODEL_NAME),
            ("Accuracy", format_metric(metrics["accuracy"])),
            ("Precision", format_metric(metrics["precision"])),
            ("Recall", format_metric(metrics["recall"])),
            ("F1 Score", format_metric(metrics["f1"])),
        ]
    )

    st.write("")

    render_metric_cards(
        [
            ("ROC-AUC", format_metric(metrics["roc_auc"])),
            ("AUPRC", format_metric(metrics["auprc"])),
            (
                "Feature Count",
                str(len(MODEL_FEATURES)) if MODEL_FEATURES else "—",
            ),
        ]
    )

    st.markdown("### Feature representation")

    st.markdown(
        """
        The source dataset contains only numerical input variables.
        `V1` through `V28` are principal components produced by PCA
        because the original transaction features cannot be disclosed
        for confidentiality reasons.

        The two non-PCA features are:

        - **Time** — seconds elapsed between each transaction and the
          first transaction in the dataset.
        - **Amount** — transaction amount.

        **Class** is the target:
        `0 = legitimate`, `1 = fraud`.
        """
    )

    if MODEL_FEATURES:
        st.markdown("### Model feature order")
        st.code(" → ".join(MODEL_FEATURES))

    st.markdown("### Model files")

    st.write(
        f"Model: `{model_path}`"
        if model_path
        else "Model file: not found"
    )

    st.write(
        f"Metadata: `{metadata_path}`"
        if metadata_path
        else "Metadata file: not found"
    )

    st.markdown("### Dataset")

    if dataset_path:
        st.write(f"Loaded dataset: `{dataset_path.name}`")
    else:
        st.write("Dataset file: not found")

    st.markdown(
        """
        ### Evaluation note

        The dataset is highly imbalanced. The original dataset
        description recommends using the Area Under the
        Precision-Recall Curve (AUPRC) because ordinary accuracy
        can be misleading for this type of classification problem.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>FraudShield AI</strong><br>
        Credit-card transaction fraud detection using machine learning.
        <br><br>
        Educational demonstration only. This application is not a
        banking, financial, fraud-investigation or transaction-approval
        system. Model predictions should not be used as the sole basis
        for real-world financial decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
