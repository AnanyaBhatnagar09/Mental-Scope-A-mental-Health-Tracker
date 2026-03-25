import json
import re
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from sentence_transformers import SentenceTransformer


TRAIN_PATH = Path("/Users/pragyabhatnagar/Desktop/mh2/train.txt")
VAL_PATH   = Path("/Users/pragyabhatnagar/Desktop/mh2/val.txt")
TEST_PATH  = Path("/Users/pragyabhatnagar/Desktop/mh2/test.txt")

_ws = re.compile(r"\s+")

def clean_text(s: str) -> str:
    return _ws.sub(" ", s.strip())


def load_json_array(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    X, y = [], []
    for row in data:
        sent = str(row.get("sentence", "")).strip()
        emo  = str(row.get("emotion", "")).strip()
        if sent and emo:
            X.append(clean_text(sent))
            y.append(emo)
    return X, y


def main():
    X_train, y_train = load_json_array(TRAIN_PATH)
    X_val,   y_val   = load_json_array(VAL_PATH)
    X_test,  y_test  = load_json_array(TEST_PATH)

    X_train_all = X_train + X_val
    y_train_all = y_train + y_val

    # 1) Embed ONCE
    sbert = SentenceTransformer("all-mpnet-base-v2")

    E_train = sbert.encode(
        X_train_all,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    E_test = sbert.encode(
        X_test,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # 2) Tune ONLY the classifier
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    grid = GridSearchCV(
        clf,
        param_grid={"C": [0.5, 1.0, 2.0, 4.0, 8.0]},
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(E_train, y_train_all)
    best_clf = grid.best_estimator_

    print("\nBest params:", grid.best_params_)

    # Evaluate
    y_pred = best_clf.predict(E_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n===== TEST METRICS =====")
    print("Accuracy       :", round(acc * 100, 2), "%")
    print("Macro Precision:", round(precision * 100, 2), "%")
    print("Macro Recall   :", round(recall * 100, 2), "%")
    print("Macro F1-Score :", round(macro_f1 * 100, 2), "%")

    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("===== CONFUSION MATRIX =====\n")
    print(confusion_matrix(y_test, y_pred))

    # Save
    np.savez("emotion_sbert_model.npz", E_train=E_train)

    import pickle
    with open("emotion_model_sbert_lr.pkl", "wb") as f:
        pickle.dump({"clf": best_clf, "sbert_model": "all-mpnet-base-v2"}, f)
    print("\nSaved: emotion_model_sbert_lr.pkl")

    # User input loop
    print("\nEnter a sentence to predict emotion (type 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            break
        emb = sbert.encode(
            [clean_text(user_input)],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        pred = best_clf.predict(emb)[0]
        print("Predicted Emotion:", pred)


if __name__ == "__main__":
    main()