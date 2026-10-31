"""Complaint routing: TF-IDF + Naive Bayes pipeline, grid search over alpha/ngrams,
MultinomialNB vs ComplementNB on imbalanced classes, confusion matrix, top words, routing threshold."""
import os
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
PHRASES = {
    "Credit Card": ["charged twice on my card", "credit limit was lowered", "interest rate increased",
                    "card statement balance wrong", "rewards points not credited", "minimum payment due",
                    "balance transfer promo", "card payment posted late"],
    "Mortgage": ["escrow account shortage", "loan modification denied", "mortgage payment misapplied",
                 "servicer foreclosure notice", "refinance closing delayed", "property tax in escrow",
                 "pmi not removed", "payment posted late to loan"],
    "Fraud": ["unauthorized transaction", "identity theft account opened", "someone stole my card",
              "scam wire transfer", "account hacked", "did not authorize this charge",
              "phishing call took my pin", "charged twice by unknown merchant"],
    "Fees": ["overdraft fee charged", "monthly maintenance fee", "late fee unfair",
             "atm fee never refunded", "wire fee too high", "fee charged without notice",
             "paper statement fee", "returned item fee"],
}
FILLER = ["i called the bank", "please help", "this is the third time", "my account", "on my statement",
          "they will not respond", "very frustrated", "customer service", "i want a refund", "since last month"]
# 🧠 ML: imbalanced like real complaint volumes - fraud is the rarest yet most urgent class
SHARES = {"Credit Card": 0.42, "Mortgage": 0.28, "Fees": 0.20, "Fraud": 0.10}


def make_complaints(n=4000):
    products, rows = list(PHRASES), []
    for product, share in SHARES.items():
        for _ in range(int(n * share)):
            words = list(rng.choice(PHRASES[product], 2)) + list(rng.choice(FILLER, 3))
            if rng.random() < 0.4:
                other = rng.choice([p for p in products if p != product])
                words.append(rng.choice(PHRASES[other]))
            label = rng.choice(products) if rng.random() < 0.06 else product
            rng.shuffle(words)
            rows.append((" ".join(words), str(label)))
    return pd.DataFrame(rows, columns=["complaint_text", "product"])


def build_search(nb_class):
    # 🧠 ML: Pipeline keeps vectoriser + model together so CV never leaks test vocabulary into training
    pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
                     ("nb", nb_class())])
    grid = {"tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 3],
            "nb__alpha": [0.01, 0.1, 0.5, 1.0]}
    # 🧠 ML: macro-F1 treats all 4 teams equally, so the small Fraud class is not ignored
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1)


def top_words(pipe, k=6):
    """For each class: words that class 'owns' (highest P(word|class) among classes), most probable first."""
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    nb = pipe.named_steps["nb"]
    logp = nb.feature_log_prob_
    if isinstance(nb, ComplementNB):
        logp = -logp  # ComplementNB stores weights of the complement, so flip the sign
    # 🧠 ML: a word belongs to the class where it is most likely; then rank by how likely it is there
    owner = logp.argmax(axis=0)
    out = {}
    for i, c in enumerate(nb.classes_):
        idx = np.where(owner == i)[0]
        out[c] = list(vocab[idx[np.argsort(logp[i, idx])[::-1][:k]]])
    return out


if __name__ == "__main__":
    df = make_complaints()
    print("Class mix:", df["product"].value_counts(normalize=True).round(2).to_dict())
    X_tr, X_te, y_tr, y_te = train_test_split(df.complaint_text, df["product"], test_size=0.25,
                                              stratify=df["product"], random_state=42)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    print(f"Baseline (always Credit Card) macro-F1: {f1_score(y_te, baseline.predict(X_te), average='macro'):.3f}\n")

    results = {}
    for name, cls in [("MultinomialNB", MultinomialNB), ("ComplementNB", ComplementNB)]:
        search = build_search(cls).fit(X_tr, y_tr)
        pred = search.predict(X_te)
        results[name] = search
        print(f"{name:<14} best CV macro-F1 {search.best_score_:.3f} | test macro-F1 "
              f"{f1_score(y_te, pred, average='macro'):.3f} | Fraud recall "
              f"{(pred[y_te == 'Fraud'] == 'Fraud').mean():.3f}")
        print(f"{'':<14} best params {search.best_params_}")

    # 🧠 ML: pick the winner by CV score (not test) to keep the test set honest
    best_name = max(results, key=lambda k: results[k].best_score_)
    best = results[best_name].best_estimator_
    pred = best.predict(X_te)
    print(f"\nChosen model: {best_name}")
    print(classification_report(y_te, pred, digits=3))

    labels = list(best.classes_)
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    print("Confusion matrix (rows = true team, columns = routed team):")
    print(cm.to_string())

    print("\nMost distinctive words per team:")
    for c, words in top_words(best).items():
        print(f"  {c:<12} {', '.join(words)}")

    # 🧠 ML: business rule - auto-route only when confident; anything mentioning fraud risk is escalated
    proba = best.predict_proba(X_te)
    conf = proba.max(axis=1)
    fraud_p = proba[:, labels.index("Fraud")]
    print("\nThreshold  auto-routed  accuracy-on-auto  sent-to-human")
    for t in [0.5, 0.6, 0.7, 0.8]:
        auto = (conf >= t) & (fraud_p < 0.3)
        acc = (pred[auto] == y_te.values[auto]).mean()
        print(f"   {t:.1f}       {auto.mean():6.1%}       {acc:6.1%}         {1 - auto.mean():6.1%}")

    path = os.path.join(tempfile.mkdtemp(), "complaint_router.joblib")
    joblib.dump(best, path)
    print(f"\nSaved pipeline -> {os.path.basename(path)}")
    print("Compliance: complaints must be handled within regulatory timelines (e.g. CFPB 15/60 days);")
    print("audit misroutes monthly and keep humans on low-confidence and possible-fraud cases.")

# ---
# Naive Bayes — HARD lab · Financial Services: Customer complaint routing to the right product team
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
