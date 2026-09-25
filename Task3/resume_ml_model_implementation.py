"""
AI-Powered Resume Intelligence - Basic ML Model Implementation
YUVA Machine Learning Engineer Internship

Problem: Predict the job category of a resume from its text.
Model: TF-IDF + Linear Support Vector Machine (LinearSVC)
"""

import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

DATA_CANDIDATES = ["UpdatedResumeDataSet.csv", "UpdatedResumeDataSet(1).csv"]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[0])
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load data
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# 2. Basic quality checks
print("Missing values:\n", df.isnull().sum())
print("Exact duplicates:", df.duplicated().sum())

# 3. Remove exact duplicates BEFORE splitting to reduce leakage.
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after duplicate removal:", df.shape)
print("Number of categories:", df["Category"].nunique())

# 4. Conservative text cleaning: preserve technical punctuation such as C++, C#, .NET and Node.js.
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_resume"] = df["Resume"].fillna("").apply(clean_text)

# 5. Stratified split for reproducibility and category representation.
X = df["clean_resume"]
y = df["Category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. TF-IDF converts resume text into a sparse numerical representation.
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

# 7. LinearSVC is suitable for high-dimensional sparse text classification.
model = LinearSVC(random_state=42)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# 8. Evaluate with accuracy and F1 scores.
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Accuracy: {accuracy:.6f}")
print(f"Macro-F1: {macro_f1:.6f}")
print(f"Weighted-F1: {weighted_f1:.6f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Confusion matrix visualization.
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(13, 11))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - Resume Category Classification")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# 10. Practical prediction function with input validation.
def predict_resume(resume_text: str) -> str:
    """Predict a category and raise a clear error for invalid input."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")
    cleaned = clean_text(resume_text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

sample_resumes = {
    "Python sample": "Python developer with Django, Flask, PostgreSQL, Pandas, NumPy and machine learning experience.",
    "Java sample": "Java developer with Spring Boot, Hibernate, JDBC, Maven and REST API experience.",
    "Network sample": "Network security engineer experienced with TCP/IP, firewalls, routing, intrusion detection and network security."
}
print("\nSample predictions:")
for name, text in sample_resumes.items():
    print(name, "->", predict_resume(text))

# 11. Error handling demonstration.
try:
    predict_resume("")
except ValueError as exc:
    print("Handled invalid input:", exc)

# 12. Persist model artifacts so inference does not require retraining.
joblib.dump(model, os.path.join(MODEL_DIR, "resume_classifier.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("Saved model artifacts in", MODEL_DIR)

# 13. Save a compact metrics file for reproducible reporting.
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Dataset after deduplication: {df.shape[0]} rows, {df.shape[1]} columns before clean_resume\n")
    f.write(f"Train rows: {len(X_train)}\nTest rows: {len(X_test)}\n")
    f.write(f"TF-IDF train shape: {X_train_tfidf.shape}\nTF-IDF test shape: {X_test_tfidf.shape}\n")
    f.write(f"Accuracy: {accuracy:.6f}\nMacro-F1: {macro_f1:.6f}\nWeighted-F1: {weighted_f1:.6f}\n")
