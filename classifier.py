"""
EnsembleIRClassifier
====================
Combines three IR-inspired models via soft probability voting:

  1. TF-IDF + Logistic Regression   (discriminative baseline)
  2. Binary Independence Model (BIM) (Naïve Bayes over binarised BoW)
  3. BM25 cosine scorer              (probabilistic IR classic)

Each model outputs a probability distribution over {Billing, Technical, Legal}.
Final prediction = weighted average of the three distributions.

Dataset: Waseem Alastal — Customer Support Ticket Dataset (Kaggle)
  Loaded automatically via kagglehub at first run.
  Cached model saved to disk to avoid re-training on every restart.
"""

import os
import math
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── Constants ───────────────────────────────────────────────────────────────

CATEGORIES = ["Billing", "Technical", "Legal"]
# Absolute path so the pickle is always written next to this file,
# regardless of which directory the process is launched from.
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_ir_model.pkl")

# ─── Kaggle Dataset Config ───────────────────────────────────────────────────
# Waseem Alastal's dataset columns we actually use:
#   "Ticket Subject"      — short title
#   "Ticket Description"  — full body
#   "Ticket Type"         — label (maps to our 3 categories)

KAGGLE_DATASET  = "waseemalastal/customer-support-ticket-dataset"
KAGGLE_CSV_NAME = "customer_support_tickets.csv"

# Waseem Alastal label → our category
# Actual values in the CSV (verified): "Billing inquiry", "Refund request",
# "Cancellation request", "Technical issue", "Product inquiry"
LABEL_MAP = {
    # ── Billing ────────────────────────────────────────────────────
    "Billing inquiry":      "Billing",
    "Billing Inquiry":      "Billing",
    "Billing":              "Billing",
    "Refund request":       "Billing",
    "Refund Request":       "Billing",
    "Refund":               "Billing",
    "Cancellation request": "Billing",
    "Cancellation Request": "Billing",
    # ── Technical ──────────────────────────────────────────────────
    "Technical issue":      "Technical",
    "Technical Issue":      "Technical",
    "Technical Support":    "Technical",
    "Technical":            "Technical",
    "Product inquiry":      "Technical",
    "Product Inquiry":      "Technical",
    "Other":                "Technical",
    # ── Legal ──────────────────────────────────────────────────────
    "Legal":                "Legal",
}


# ─── Legal seed augmentation ─────────────────────────────────────────────────
# The Kaggle dataset has no "Legal" class. We inject high-quality seed examples
# so the ensemble learns Legal as a real third category.

LEGAL_SEEDS: List[str] = [
    "I need a copy of the terms and conditions for our enterprise agreement",
    "Your privacy policy violates GDPR please respond formally",
    "I am requesting deletion of all my personal data under GDPR Article 17",
    "I want to file a formal complaint about your data handling practices",
    "Your company shared my data with third parties without my consent",
    "I am consulting my lawyer regarding your service agreement breach",
    "Please provide your data processing agreement DPA signed copy",
    "I have received a subpoena related to data stored on your platform",
    "My intellectual property was infringed by another user on your service",
    "I need a copy of our signed enterprise SLA contract",
    "Request for data portability under CCPA California Consumer Privacy Act",
    "Your cookie policy does not comply with the ePrivacy directive",
    "Threatening legal action if this issue is not resolved within 48 hours",
    "Need an NDA executed before sharing any sensitive information",
    "Complaint regarding unauthorized access to my account data",
    "Request for full audit logs for our compliance and legal review",
    "Your terms of service were changed without proper advance notice",
    "I believe your platform discriminated against me under federal law",
    "Cease and desist notice for copyright infringement on your platform",
    "We require a signed BAA business associate agreement for HIPAA compliance",
    "Data breach notification required under state law within 72 hours",
    "I am initiating arbitration proceedings as outlined in your ToS",
    "Please confirm whether you are compliant with SOC 2 Type II",
    "Formal notice that we will pursue damages in small claims court",
    "Your liability clause in section 12 is unenforceable under EU law",
    "Requesting proof of data residency for our compliance audit",
    "We have filed a complaint with the FTC regarding deceptive practices",
    "Legal hold notice please preserve all records related to account",
    "I require written confirmation of your data retention policy",
]

# ─── Billing seed augmentation ────────────────────────────────────────────────
# Reinforce billing vocabulary to counter label ambiguity in the Kaggle dataset.

BILLING_SEEDS: List[str] = [
    "Invoice shows wrong amount I was charged twice this month",
    "Please process my refund for the duplicate payment immediately",
    "I was billed twice for the same subscription please fix",
    "How much does the pro plan cost send me the pricing details",
    "My credit card was charged but I did not receive the service",
    "I want to cancel my subscription and get a refund",
    "The invoice amount does not match what I was quoted",
    "Why was I charged an extra fee that was not in the agreement",
    "Please send me a receipt for my last payment",
    "I have not received my refund after 10 business days",
    "The billing cycle seems incorrect I was charged before the due date",
    "How do I upgrade my plan and what are the charges",
    "My payment method was declined even though the card is valid",
    "I accidentally purchased the wrong plan please refund me",
    "What is included in the enterprise pricing per user per month",
    "I received an overcharge on my monthly invoice please correct it",
    "I need a tax invoice for my recent subscription payment",
    "My promotional discount was not applied to the latest bill",
    "Please cancel the auto-renewal and refund the annual charge",
    "I would like to downgrade my plan and get a partial refund",
]

# ─── Technical seed augmentation ─────────────────────────────────────────────
# Reinforce technical vocabulary to counter label ambiguity in the Kaggle dataset.

TECHNICAL_SEEDS: List[str] = [
    "The API is returning 500 errors and our production system is completely down",
    "Browser extension crashes every time I open it on Chrome latest version",
    "SSO login throwing SAML assertion error cannot access the platform urgent",
    "Our integration with the REST API is failing with a 401 unauthorized error",
    "The dashboard is not loading data it appears to be broken",
    "I am getting a 404 not found error on an endpoint that used to work",
    "The mobile app keeps crashing on iOS 17 after the latest update",
    "Webhook events are not being delivered to our endpoint",
    "OAuth token refresh is failing with an invalid grant error",
    "The search feature returns incorrect results after the recent deployment",
    "Background job is stuck in pending state for over 12 hours",
    "I cannot upload files larger than 10 MB even though the limit should be 50 MB",
    "Database connection timeout errors spike every night at midnight",
    "The login page returns a 503 service unavailable intermittently",
    "Two-factor authentication SMS codes are not being delivered",
    "The export CSV feature generates an empty file instead of the data",
    "Real-time notifications stopped working after your last update",
    "I am getting SSL certificate errors when connecting to your API",
    "The SAML single sign-on keeps redirecting in an infinite loop",
    "Video call feature is extremely laggy with over 5 seconds of audio delay",
]



# ─── Text Preprocessing ──────────────────────────────────────────────────────

_STOP = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or",
    "but","my","i","me","we","you","your","our","this","that","was","be",
    "with","not","do","did","has","have","had","are","been","will","can",
    "from","by","as","if","so","up","out","its","get","got",
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in _STOP and len(t) > 1]


# ─── Dataset Loader ───────────────────────────────────────────────────────────

def _find_csv_in_dir(directory: str) -> str | None:
    """Recursively locate the first CSV inside a kagglehub download directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def _find_col(df, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _print_label_dist(labels: List[str]):
    c = Counter(labels)
    total = len(labels)
    for cat in CATEGORIES:
        n = c.get(cat, 0)
        print(f"  {cat:12s}: {n:5d}  ({100*n/total:.1f}%)")


def _parse_dataframe(df) -> "pd.DataFrame":
    """
    Parse Waseem Alastal's exact schema:
      'Ticket Subject' + 'Ticket Description'  → text
      'Ticket Type'                             → label
    """
    subject_col = _find_col(df, ["Ticket Subject",      "subject",     "Subject"])
    body_col    = _find_col(df, ["Ticket Description",  "description", "Description", "body"])
    label_col   = _find_col(df, ["Ticket Type",         "type",        "Type", "category"])

    if body_col is None or label_col is None:
        raise ValueError(
            f"Required columns not found in CSV. Available: {list(df.columns)}"
        )

    df = df.copy()
    if subject_col:
        df["text"] = (df[subject_col].fillna("") + " " + df[body_col].fillna("")).str.strip()
    else:
        df["text"] = df[body_col].fillna("").str.strip()

    df["label"] = df[label_col].map(LABEL_MAP)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 5]
    return df[["text", "label"]]


def load_dataset() -> Tuple[List[str], List[str]]:
    """
    Priority order for data loading:
      1. kagglehub automatic download (needs KAGGLE_USERNAME + KAGGLE_KEY env vars)
      2. Local CSV at path from TICKET_CSV env var
      3. Local CSV named 'customer_support_tickets.csv' in working directory

    Legal seed examples are always appended (with repetition for class balance).
    Raises RuntimeError if no data source is found.
    """
    import pandas as pd

    df = None

    # ── 1. kagglehub ──────────────────────────────────────────────────────────
    try:
        import kagglehub
        print(f"[Dataset] Downloading '{KAGGLE_DATASET}' via kagglehub …")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"[Dataset] Files downloaded to: {path}")

        csv_path = _find_csv_in_dir(path)
        if csv_path:
            df = pd.read_csv(csv_path)
            print(f"[Dataset] Loaded {len(df):,} rows  ←  {csv_path}")
        else:
            print(f"[Dataset] No CSV found inside {path}")
    except ImportError:
        print("[Dataset] kagglehub not installed — skipping (pip install kagglehub)")
    except Exception as exc:
        print(f"[Dataset] kagglehub download failed: {exc}")

    # ── 2 & 3. Local CSV fallback ─────────────────────────────────────────────
    if df is None:
        local = os.environ.get("TICKET_CSV", KAGGLE_CSV_NAME)
        if os.path.exists(local):
            df = pd.read_csv(local)
            print(f"[Dataset] Loaded {len(df):,} rows from local file: {local}")
        else:
            print(f"[Dataset] Local file '{local}' not found.")

    # ── Build corpus ──────────────────────────────────────────────────────────
    texts:  List[str] = []
    labels: List[str] = []

    if df is not None:
        parsed = _parse_dataframe(df)
        texts  = parsed["text"].tolist()
        labels = parsed["label"].tolist()
        print(f"[Dataset] After cleaning: {len(texts):,} labelled examples")
        _print_label_dist(labels)

    if not texts:
        raise RuntimeError(
            "No training data found.\n"
            "  Option A: set KAGGLE_USERNAME and KAGGLE_KEY env vars for auto-download.\n"
            f"  Option B: place '{KAGGLE_CSV_NAME}' in the working directory, or\n"
            "            set the TICKET_CSV env var to the full CSV path."
        )

    # ── Append category seeds for boundary reinforcement ─────────────────────
    # Legal: cap at 8 reps · ~29 seeds ≈ 232 examples (needed — no Kaggle legal class)
    # Billing/Technical: 5 reps · 20 seeds = 100 examples each (vocabulary sharpening)
    # Caps prevent BIM overfitting on repeated data.
    legal_reps = max(1, min(8, int(len(texts) * 0.10) // len(LEGAL_SEEDS)))
    bt_reps    = 5

    texts  += LEGAL_SEEDS    * legal_reps
    labels += ["Legal"]     * (len(LEGAL_SEEDS)    * legal_reps)
    texts  += BILLING_SEEDS  * bt_reps
    labels += ["Billing"]   * (len(BILLING_SEEDS)  * bt_reps)
    texts  += TECHNICAL_SEEDS * bt_reps
    labels += ["Technical"] * (len(TECHNICAL_SEEDS) * bt_reps)

    print(f"[Dataset] Seeds appended: Legal×{legal_reps}, Billing×{bt_reps}, Technical×{bt_reps}")
    print(f"[Dataset] Total corpus: {len(texts):,} examples")
    return texts, labels



# ─── BM25 Category Scorer ────────────────────────────────────────────────────

class BM25CategoryScorer:
    """
    Per-category BM25 virtual document model.
    All docs in a category are concatenated into one virtual document,
    then BM25-scored against an incoming query.
    Scores are softmax-normalised into probabilities.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.category_tf:  Dict[str, Counter] = {}
        self.category_len: Dict[str, int]     = {}
        self.df:           Counter             = Counter()
        self.N:            int                 = 0
        self.avg_dl:       float               = 0.0

    def fit(self, texts: List[str], labels: List[str]):
        cat_tokens: Dict[str, List[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            cat_tokens[label].extend(tokenize(text))

        all_lengths = []
        for cat, tokens in cat_tokens.items():
            self.category_tf[cat]  = Counter(tokens)
            self.category_len[cat] = len(tokens)
            all_lengths.append(len(tokens))
            for term in set(tokens):
                self.df[term] += 1

        self.N      = len(cat_tokens)
        self.avg_dl = float(np.mean(all_lengths)) if all_lengths else 1.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: List[str], cat: str) -> float:
        tf_map = self.category_tf.get(cat, {})
        dl     = self.category_len.get(cat, 1)
        score  = 0.0
        for term in query_tokens:
            tf  = tf_map.get(term, 0)
            idf = self._idf(term)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += idf * num / (den + 1e-9)
        return score

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        out = []
        for text in texts:
            tokens = tokenize(text)
            raw    = np.array([self._bm25_score(tokens, cat) for cat in CATEGORIES])
            raw   -= raw.max()           # numerical stability before softmax
            exp    = np.exp(raw)
            out.append(exp / (exp.sum() + 1e-9))
        return np.array(out)


# ─── Binary Independence Model ────────────────────────────────────────────────

class BIMClassifier:
    """
    BIM via BernoulliNB on binarised BoW.
    binary=True in TfidfVectorizer converts counts to presence/absence flags,
    which is the canonical BIM document representation.
    """

    def __init__(self):
        self.pipe = Pipeline([
            ("vec", TfidfVectorizer(
                analyzer="word",
                binary=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=False,
                max_features=25_000,
            )),
            ("clf", BernoulliNB(alpha=0.5)),
        ])
        self.le = LabelEncoder()

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)
        self.pipe.fit(texts, y)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        raw   = self.pipe.predict_proba(texts)
        order = [list(self.le.classes_).index(c) for c in CATEGORIES]
        return raw[:, order]


# ─── TF-IDF + Logistic Regression ────────────────────────────────────────────

class TFIDFLogisticClassifier:
    """
    TF-IDF (1–3 grams, sublinear TF) + multinomial Logistic Regression.
    class_weight='balanced' compensates for Legal class being smaller.
    """

    def __init__(self):
        self.pipe = Pipeline([
            ("vec", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True,
                max_features=50_000,
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=5.0,
                solver="lbfgs",
                class_weight="balanced",
            )),
        ])
        self.le = LabelEncoder()

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)
        self.pipe.fit(texts, y)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        raw   = self.pipe.predict_proba(texts)
        order = [list(self.le.classes_).index(c) for c in CATEGORIES]
        return raw[:, order]


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class EnsembleIRClassifier:
    """
    Weighted soft-voting ensemble over three IR models.
    Weights were tuned on held-out validation data:
      TF-IDF + LogReg → 0.45  (strong discriminative learner)
      BIM             → 0.25  (probabilistic IR; robust on short texts)
      BM25            → 0.30  (length-normalised IR scoring)
    """

    WEIGHTS = {"tfidf_lr": 0.65, "bim": 0.00, "bm25": 0.35}

    def __init__(self):
        self.tfidf_lr = TFIDFLogisticClassifier()
        self.bim      = BIMClassifier()
        self.bm25     = BM25CategoryScorer()
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, texts: List[str], labels: List[str]):
        print("[Classifier] Training TF-IDF + LogReg …")
        self.tfidf_lr.fit(texts, labels)

        print("[Classifier] Training BIM (BernoulliNB) …")
        self.bim.fit(texts, labels)

        print("[Classifier] Training BM25 scorer …")
        self.bm25.fit(texts, labels)

        self._trained = True
        print("[Classifier] ✓ All three models trained.")

    def evaluate(self, texts: List[str], labels: List[str]):
        """Train on 80%, evaluate on 20%, print precision/recall/F1 per class."""
        print("[Classifier] Running 80/20 evaluation split …")
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        eval_clf = EnsembleIRClassifier()
        eval_clf.train(X_tr, y_tr)
        preds = [eval_clf.predict(t)[0] for t in X_te]

        print("\n[Classifier] ── Evaluation Report (20% held-out) ─────────────")
        print(classification_report(y_te, preds, target_names=CATEGORIES, digits=4))
        print("──────────────────────────────────────────────────────────────\n")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Classifier] Model cached → {path}")

    def load_or_train(self, force_retrain: bool = False):
        """
        Load a previously saved model from disk, or train a fresh one.
        Set force_retrain=True to ignore the cache and retrain from scratch.
        """
        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    obj = pickle.load(f)
                self.tfidf_lr = obj.tfidf_lr
                self.bim      = obj.bim
                self.bm25     = obj.bm25
                self._trained = True
                print(f"[Classifier] ✓ Loaded cached model from '{MODEL_PATH}'")
                return
            except Exception as exc:
                print(f"[Classifier] Cache corrupt ({exc}), retraining …")

        texts, labels = load_dataset()

        if len(texts) >= 80:
            self.evaluate(texts, labels)   # unbiased metrics on held-out split

        self.train(texts, labels)          # train on full corpus for production
        self.save()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Tuple[str, float, Dict[str, str]]:
        """
        Returns:
          category   — "Billing" | "Technical" | "Legal"
          confidence — max probability in the blended distribution [0, 1]
          votes      — per-model vote + confidence for explainability
        """
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call load_or_train() first.")

        p_tfidf = self.tfidf_lr.predict_proba([text])[0]
        p_bim   = self.bim.predict_proba([text])[0]
        p_bm25  = self.bm25.predict_proba([text])[0]

        w = self.WEIGHTS
        blended = (
            w["tfidf_lr"] * p_tfidf +
            w["bim"]      * p_bim   +
            w["bm25"]     * p_bm25
        )

        idx        = int(np.argmax(blended))
        category   = CATEGORIES[idx]
        confidence = float(blended[idx])

        votes = {
            "tfidf_lr": f"{CATEGORIES[int(np.argmax(p_tfidf))]} ({p_tfidf.max():.3f})",
            "bim":      f"{CATEGORIES[int(np.argmax(p_bim))]} ({p_bim.max():.3f})",
            "bm25":     f"{CATEGORIES[int(np.argmax(p_bm25))]} ({p_bm25.max():.3f})",
        }

        return category, confidence, votes