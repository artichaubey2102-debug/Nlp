# ============================================
# 🌈 Streamlit NLP Phase-wise Model Comparison
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Feature Extractors
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents]
    return f"{len(sents)} {' '.join([s.split()[0] for s in sents if s])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase-wise Analysis", layout="wide")
st.title("🧠 NLP Phase-wise Analysis")
st.markdown(
    "<p style='color:gray;'>Upload a dataset, choose an NLP phase, and compare multiple ML models.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.write("---")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Select Text Column", df.columns)
    with col2:
        target_col = st.selectbox("Select Target Column", df.columns)

    phase = st.selectbox(
        "🔎 Choose NLP Phase",
        ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
    )

    if st.button("🚀 Run Model Comparison"):
        X = df[text_col].astype(str)
        y = df[target_col]

        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                      columns=["polarity", "subjectivity"])

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        else:  # Pragmatic
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                      columns=pragmatic_words)

        results = evaluate_models(X_features, y)
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df[results_df["Accuracy"].apply(lambda x: isinstance(x, (int,float)))]
        results_df = results_df.sort_values(by="Accuracy", ascending=False)

        st.subheader("🏆 Model Accuracy")
        st.table(results_df)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.bar(results_df["Model"], results_df["Accuracy"], color="#4CAF50", alpha=0.8)
        plt.ylabel("Accuracy (%)")
        plt.title(f"Performance on {phase}")
        for i, v in enumerate(results_df["Accuracy"]):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        st.pyplot(plt)
else:
    st.info("⬅️ Please upload a CSV file to start.")
