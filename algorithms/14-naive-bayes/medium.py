import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

rng = np.random.default_rng(42)

# --- 1. Generate 3,000 synthetic consumer complaints (CFPB-style narratives) ---
# 🐍 PY: a dict of product -> list of phrases typical for that product
phrases = {
    "Credit Card": ["charged twice on my card", "credit limit was lowered", "interest rate increased",
                    "card statement balance wrong", "rewards points not credited", "minimum payment"],
    "Mortgage": ["escrow account shortage", "loan modification denied", "mortgage payment misapplied",
                 "servicer foreclosure notice", "refinance closing delayed", "property tax in escrow"],
    "Fraud": ["unauthorized transaction", "identity theft account opened", "someone stole my card",
              "scam wire transfer", "account hacked", "did not authorize this charge"],
    "Fees": ["overdraft fee charged", "monthly maintenance fee", "late fee unfair",
             "atm fee refunded never", "wire fee too high", "fee charged without notice"],
}
# 🧠 ML: shared filler words appear in every class - real complaints are full of them
filler = ["i called the bank", "please help", "this is the third time", "my account",
          "they will not respond", "very frustrated", "customer service", "on my statement"]
# 🧠 ML: real complaints mix topics (a fee on a card, fraud on a mortgage) - this keeps the task honest
products = list(phrases)

rows = []
# 🐍 PY: zip pairs each class with its share of the complaint volume
for product, share in zip(products, [0.35, 0.25, 0.20, 0.20]):
    for _ in range(int(3000 * share)):
        words = list(rng.choice(phrases[product], 2)) + list(rng.choice(filler, 3))
        if rng.random() < 0.4:
            other = rng.choice([p for p in products if p != product])
            words.append(rng.choice(phrases[other]))
        # 🧠 ML: about 8% of consumers pick the wrong product on the form - label noise
        label = rng.choice(products) if rng.random() < 0.08 else product
        # 🐍 PY: shuffle in place, then " ".join glues the phrases into one narrative string
        rng.shuffle(words)
        rows.append({"complaint_text": " ".join(words), "product": label})
df = pd.DataFrame(rows)
print(df.sample(3, random_state=1).to_string(index=False))
print("\nClass counts:\n" + df["product"].value_counts().to_string())

# --- 2. Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df["complaint_text"], df["product"], test_size=0.25, stratify=df["product"], random_state=42)

# --- 3. Turn text into word counts ---
# 🧠 ML: bag-of-words - each complaint becomes a row of counts, one column per vocabulary word
vectorizer = CountVectorizer(stop_words="english")
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)
print(f"\nVocabulary: {len(vectorizer.vocabulary_)} words; matrix shape {Xtr.shape}")

# --- 4. Train Multinomial Naive Bayes ---
# 🧠 ML: alpha=1.0 is Laplace smoothing, same idea as the hand-made easy level
nb = MultinomialNB(alpha=1.0)
nb.fit(Xtr, y_train)
pred = nb.predict(Xte)
print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, digits=3))

# --- 5. Route new complaints ---
new = ["someone opened a card in my name identity theft",
       "they charged an overdraft fee twice this month",
       "my escrow went up and the mortgage payment doubled"]
# 🐍 PY: predict_proba returns one row per complaint; argmax picks the most likely class index
proba = nb.predict_proba(vectorizer.transform(new))
for text, p in zip(new, proba):
    print(f"{nb.classes_[p.argmax()]:<12} {p.max():.0%}  <- '{text}'")

# 🧠 ML: confidence-based routing - low-confidence cases go to a human triage queue
test_conf = nb.predict_proba(Xte).max(axis=1)
auto = test_conf >= 0.90
print(f"\nAuto-route {auto.mean():.0%} of complaints at >=90% confidence; "
      f"accuracy on those {accuracy_score(y_test[auto], pred[auto]):.1%}")

# ---
# Naive Bayes — MEDIUM lab · Financial Services: Customer complaint routing to the right product team
# Part of The Architect's ML Playbook: https://github.com/EdwardSuresh/architects-ml-playbook
# Legend: '# 🐍 PY:' = Python lesson, '# 🧠 ML:' = machine-learning lesson
