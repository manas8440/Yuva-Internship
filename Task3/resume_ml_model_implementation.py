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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

DATA_CANDIDATES = ["UpdatedResumeDataSet(3).csv", "UpdatedResumeDataSet.csv", "UpdatedResumeDataSet(1).csv"]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[0])
MODEL_DIR = "models_authentic"
RESULTS_DIR = "results_authentic"
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

# 10.1 Dataset-backed practical demonstration.
# Use actual held-out records from the supplied dataset rather than hand-written
# sample resumes. These records were NOT used to fit the classifier.
test_records = df.loc[X_test.index, ["Category", "Resume"]].copy()
test_records["predicted_category"] = y_pred
test_records["correct"] = test_records["Category"] == test_records["predicted_category"]

# LinearSVC decision margin is NOT a probability. It is the difference between
# the highest and second-highest decision scores for the predicted classes.
decision_scores = model.decision_function(X_test_tfidf)
if decision_scores.ndim == 1:
    decision_scores = decision_scores.reshape(-1, 1)
sorted_scores = np.sort(decision_scores, axis=1)
test_records["decision_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2] if decision_scores.shape[1] > 1 else np.nan
test_records["resume_excerpt"] = (
    test_records["Resume"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
)

# Select a deterministic set of actual held-out examples from different
# categories. If a category is absent from the test partition, it is skipped.
demo_categories = [
    "Data Science", "Java Developer", "Python Developer",
    "Network Security Engineer", "DevOps Engineer", "Testing",
    "Web Designing", "Business Analyst"
]
demo_parts = []
for category in demo_categories:
    part = test_records[test_records["Category"] == category]
    if not part.empty:
        demo_parts.append(part.iloc[[0]])
dataset_demo = pd.concat(demo_parts) if demo_parts else test_records.head(8)

print("\nDataset-backed practical demonstration:")
print(dataset_demo[
    ["Category", "predicted_category", "correct", "decision_margin", "resume_excerpt"]
].to_string(index=False))

dataset_demo.to_csv(
    os.path.join(RESULTS_DIR, "practical_demonstration.csv"),
    index=False
)

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

# 11. Practical prediction function with input validation.
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

# 12. Error handling demonstration.
try:
    predict_resume("")
except ValueError as exc:
    print("Handled invalid input:", exc)

# 13. Persist model artifacts so inference does not require retraining.
joblib.dump(model, os.path.join(MODEL_DIR, "resume_classifier.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("Saved model artifacts in", MODEL_DIR)



# 15. Practical testing and debugging checks.
# These checks do not retrain the model or change the evaluation result.
def run_verification_checks():
    print("\nTesting and debugging checks:")

    # Test 1: Text-cleaning behavior.
    cleaned = clean_text("  Python\nDeveloper   with Django  ")
    assert cleaned == "python developer with django"
    print("PASS - text cleaning normalizes case and whitespace")

    # Test 2: Train/test feature dimensions must match.
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1]
    assert X_train_tfidf.shape[1] == len(tfidf.vocabulary_)
    print("PASS - train/test TF-IDF dimensions are consistent")

    # Test 3: Predictions must belong to the learned category set.
    known_categories = set(model.classes_)
    assert set(y_pred).issubset(known_categories)
    print("PASS - all predicted labels belong to known categories")

    # Test 4: Empty and non-string input must be rejected clearly.
    for invalid_input in ["", "   ", None, 123]:
        try:
            predict_resume(invalid_input)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input was not rejected: {invalid_input!r}")
    print("PASS - invalid/empty resume inputs are rejected")

    # Test 5: Representative inference should return the expected category
    # for the three documented examples used in the practical demonstration.
    expected_categories = {
        "Python sample": "Python Developer",
        "Java sample": "Java Developer",
        "Network sample": "Network Security Engineer",
    }
    for name, sample_text in sample_resumes.items():
        prediction = predict_resume(sample_text)
        assert prediction == expected_categories[name], (
            f"Unexpected prediction for {name}: {prediction}"
        )
    print("PASS - representative sample predictions match documented outputs")

    # Test 6: Saved artifacts must be reloadable without retraining.
    loaded_model = joblib.load(os.path.join(MODEL_DIR, "resume_classifier.pkl"))
    loaded_tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    loaded_prediction = loaded_model.predict(
        loaded_tfidf.transform([clean_text(sample_resumes["Python sample"])])
    )[0]
    assert loaded_prediction == expected_categories["Python sample"]
    print("PASS - saved model and vectorizer reload correctly")

    print("All verification checks passed.")

run_verification_checks()

# Save a testing summary for auditability.
with open(os.path.join(RESULTS_DIR, "testing_results.txt"), "w", encoding="utf-8") as f:
    f.write("Text cleaning: PASS\n")
    f.write("TF-IDF train/test dimensions: PASS\n")
    f.write("Predicted labels in known category set: PASS\n")
    f.write("Invalid/empty input handling: PASS\n")
    f.write("Representative sample predictions: PASS\n")
    f.write("Saved artifact reload: PASS\n")

# 14. Save a compact metrics file for reproducible reporting.
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Dataset after deduplication: {df.shape[0]} rows, {df.shape[1]} columns before clean_resume\n")
    f.write(f"Train rows: {len(X_train)}\nTest rows: {len(X_test)}\n")
    f.write(f"TF-IDF train shape: {X_train_tfidf.shape}\nTF-IDF test shape: {X_test_tfidf.shape}\n")
    f.write(f"Accuracy: {accuracy:.6f}\nMacro-F1: {macro_f1:.6f}\nWeighted-F1: {weighted_f1:.6f}\n")
    f.write(f"Dataset-backed practical demonstration rows: {len(dataset_demo)}\n")
