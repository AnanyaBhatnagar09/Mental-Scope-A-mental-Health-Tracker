import json
import re
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

# ---------------- CONFIG ----------------
DATA_PATH = "intents chatbot nd4.json"
MODEL_OUT = "intent_model_best_final.joblib"

MIN_SAMPLES_PER_INTENT = 5
CONF_THRESHOLD = 0.30
FACT_TAG = "info_query"
# ---------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

MERGE = {
    # 1. DISTRESS (Negative emotional states, Crisis, Death)
    "sad": "distress",
    "stressed": "distress",
    "depressed": "distress",
    "anxious": "distress",
    "scared": "distress",
    "worthless": "distress",
    "anger": "distress",
    "overthinking": "distress",
    "motivation-low": "distress",
    "relationship-issue": "distress",
    "exam-pressure": "distress",
    "death": "distress",
    "crisis": "distress",
    "suicide": "distress",
    "social": "distress",
    "friends": "distress",

    # 2. WELLNESS_COPING (Positive states, meditation, coping, advice)
    "meditation": "wellness_coping",
    "user-meditation": "wellness_coping",
    "breathing-exercise": "wellness_coping",
    "sleep": "wellness_coping",
    "happy": "wellness_coping",
    "user-agree": "wellness_coping",
    "understand": "wellness_coping",
    "affirmation": "wellness_coping",
    "thanks": "wellness_coping",
    "coping-suggestions": "wellness_coping",
    "user-tried-approach": "wellness_coping",
    "user-advice": "wellness_coping",
    "pandora-useful": "wellness_coping",
    "coping": "wellness_coping",
    "wellness": "wellness_coping",
    "coping_advice": "wellness_coping",

    # 3. INFO_QUERY (Facts and Bot queries)
    "fact": "info_query",
    "learn": "info_query",
    "learn-mental-health": "info_query",
    "learn-more": "info_query",
    "mental-health-fact": "info_query",
    "ask": "info_query",
    "about": "info_query",
    "problem": "info_query",
    "location": "info_query",
    "help": "info_query",
    "info_query": "info_query",

    # 4. SOCIAL_FALLBACK (Social talk, humor, greeting, fallback, utility)
    "morning": "social_fallback",
    "night": "social_fallback",
    "afternoon": "social_fallback",
    "evening": "social_fallback",
    "casual": "social_fallback",
    "goodbye": "social_fallback",
    "greeting": "social_fallback",
    "jokes": "social_fallback",
    "humor": "social_fallback",
    "default": "social_fallback",
    "no-approach": "social_fallback",
    "something-else": "social_fallback",
    "not-talking": "social_fallback",
    "neutral-response": "social_fallback",
    "fallback": "social_fallback",
    "repeat": "social_fallback",
    "done": "social_fallback",
    "no-response": "social_fallback",
    "hate-me": "social_fallback",
    "hate-you": "social_fallback",
    "stupid": "social_fallback",
    "wrong": "social_fallback",
    "negative": "social_fallback",
    "utility": "social_fallback",
    "greeting_humor": "social_fallback",
    "fallback_utility": "social_fallback",
}


# ---------------- DATA ----------------
def is_valid_pattern(p: str) -> bool:
    if not p or len(p.strip()) < 2:
        return False
    if re.fullmatch(r"[\W_]+", p):
        return False
    return any(ch.isalnum() for ch in p)


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        intents = json.load(f)

    texts, labels = [], []
    for intent in intents["intents"]:
        tag = intent.get("tag", "").strip()

        if re.fullmatch(r"fact-\d+", tag):
            tag = FACT_TAG

        tag = MERGE.get(tag, tag)

        for p in intent.get("patterns", []):
            if isinstance(p, str) and is_valid_pattern(p):
                cleaned = clean_text(p)
                if cleaned:
                    texts.append(cleaned)
                    labels.append(tag)

    return texts, labels



# ---------------- MODEL ----------------
def build_model():
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 3), 
        min_df=2, # Capturing more specific patterns
        sublinear_tf=True,
    )

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )

    base = LinearSVC(
        loss="squared_hinge",
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
        dual=True
    )

    return Pipeline([
        ("features", FeatureUnion([
            ("word", tfidf_word),
            ("char", tfidf_char),
        ])),
        ("svm", base)
    ])


# ---------------- TRAIN ----------------
def main():
    texts, labels = load_data(DATA_PATH)

    counts = Counter(labels)
    removed = {c for c, n in counts.items() if n < MIN_SAMPLES_PER_INTENT}

    texts = [t for t, y in zip(texts, labels) if y not in removed]
    labels = [y for y in labels if y not in removed]

    counts = Counter(labels)
    print("\nSamples per intent:")
    for k, v in sorted(counts.items(), key=lambda x: x[1]):
        print(f"{k:20s} {v}")
    print("Total intents:", len(counts))

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2, 
        stratify=labels,
        random_state=42
    )

    # Targeted Grid Search
    # Granular search
    pipeline = build_model()
    param_grid = {
        'svm__C': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0],
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    clf_base = grid.best_estimator_

    # Calibration
    train_min = min(Counter(y_train).values())
    cv_calib = 3 if train_min >= 3 else 2
    
    clf = CalibratedClassifierCV(
        clf_base,
        cv=cv_calib,
        method="sigmoid"
    )
    clf.fit(X_train, y_train)

    # ---------------- METRICS ----------------
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    macro_precision = precision_score(y_test, test_preds, average="macro", zero_division=0)
    weighted_precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)

    macro_recall = recall_score(y_test, test_preds, average="macro", zero_division=0)
    weighted_recall = recall_score(y_test, test_preds, average="weighted", zero_division=0)

    macro_f1 = f1_score(y_test, test_preds, average="macro")
    weighted_f1 = f1_score(y_test, test_preds, average="weighted")

    print(f"\n📊 Training Accuracy:      {train_acc*100:.2f}%")
    print(f"📊 Test Accuracy:          {test_acc*100:.2f}%")
    print(f"📊 Macro Precision:        {macro_precision*100:.2f}%")
    print(f"📊 Weighted Precision:     {weighted_precision*100:.2f}%")
    print(f"📊 Macro Recall:           {macro_recall*100:.2f}%")
    print(f"📊 Weighted Recall:        {weighted_recall*100:.2f}%")
    print(f"📊 Macro F1-score:         {macro_f1*100:.2f}%")
    print(f"📊 Weighted F1-score:      {weighted_f1*100:.2f}%")

    # Final Cross-Val
    cv_f1 = cross_val_score(clf, texts, labels, cv=skf, scoring="f1_macro")
    print(f"\n📊 Cross-Val Macro F1:     {cv_f1.mean()*100:.2f}% (± {cv_f1.std()*100:.2f}%)")

    print("\nClassification report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    joblib.dump({
        "pipeline": clf,
        "merge_map": MERGE,
        "confidence_threshold": CONF_THRESHOLD,
        "fact_tag": FACT_TAG,
        "clean_text": clean_text
    }, MODEL_OUT)

    print(f"\n✅ Saved improved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()