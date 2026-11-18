import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz


def clean_text(value):
    text = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def canonical_supplier(value):
    text = clean_text(value).upper()
    text = re.sub(r"\b(PTY|LTD|LIMITED|LLC|INC|SA|GMBH|CO|CORP)\b", "", text)
    return re.sub(r"\s+", " ", text).strip() or "UNKNOWN"


def generate_keys(df, text_col, supplier_col, training_df=None, supplier_training=None):
    df = df.copy()
    training_df = None if training_df is None else training_df.copy()

    if supplier_col in df:
        supplier_names = df[supplier_col].fillna("UNKNOWN").astype(str)
    else:
        supplier_names = pd.Series("UNKNOWN", index=df.index)

    if supplier_training is not None and "supplier_key" in supplier_training and "supplier_name" in supplier_training:
        references = supplier_training.dropna(subset=["supplier_name"])
        vendor_to_key = {}
        name_to_key = {}
        for _, row in references.iterrows():
            key = str(row["supplier_key"])
            name_to_key.setdefault(canonical_supplier(row["supplier_name"]), key)
            if "vendor_id" in row and pd.notna(row["vendor_id"]):
                vendor_to_key[str(row["vendor_id"])] = key
        keys = []
        for _, value in supplier_names.items():
            raw = str(value)
            key = vendor_to_key.get(raw)
            if key is None:
                canon = canonical_supplier(raw)
                key = name_to_key.get(canon)
                if key is None:
                    candidates = [(fuzz.ratio(canon, name), k) for name, k in name_to_key.items()]
                    if candidates:
                        score, candidate = max(candidates)
                        key = candidate if score >= 82 else "SUP-" + canon[:24].replace(" ", "-")
                    else:
                        key = "SUP-" + canon[:24].replace(" ", "-")
            keys.append(key)
        df["supplier_key"] = keys
    else:
        df["supplier_key"] = supplier_names.map(canonical_supplier).map(lambda x: "SUP-" + x[:40].replace(" ", "-"))

    descriptions = df[text_col].fillna("UNKNOWN").astype(str).map(clean_text)
    if training_df is not None and "item_key" in training_df:
        train = training_df.dropna(subset=["item_desc", "item_key"])
        train_text = train["item_desc"].fillna("UNKNOWN").astype(str).map(clean_text)
        if len(train):
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=6000)
            train_matrix = vectorizer.fit_transform(train_text)
            query_matrix = vectorizer.transform(descriptions)
            nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_matrix)
            distances, indices = nn.kneighbors(query_matrix)
            keys = []
            for distance, index in zip(distances[:, 0], indices[:, 0]):
                keys.append(str(train.iloc[index]["item_key"]) if distance < 0.55 else "ITEM-" + descriptions.iloc[len(keys)][:32].replace(" ", "-"))
            df["item_key"] = keys
            return df

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000)
    matrix = vectorizer.fit_transform(descriptions)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    df["item_key"] = ["ITEM-" + str(abs(hash(clean_text(d))) % 100000000) for d in descriptions]
    return df
