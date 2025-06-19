# src/common/preprocessing.py

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import joblib

class TextCleaner(TransformerMixin):
    """
    Transformer para limpeza básica de texto:
      - lowercase
      - remoção de pontuação
    """
    def __init__(self):
        self.regex = re.compile(r"[^\w\s]", re.UNICODE)

    def transform(self, X, y=None):
        # X é uma lista ou Series de textos
        return X.astype(str)\
                .str.lower()\
                .apply(lambda s: self.regex.sub("", s))

    def fit(self, X, y=None):
        return self

def load_datasets(spam_path, phishing_path):
    # SMS Spam Collection (UCI)
    df_spam = pd.read_csv(
        spam_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin1"
    )
    # SMS Phishing: ajuste o nome das colunas conforme seu CSV
    df_phish = pd.read_csv(
        phishing_path,
        sep=",",
        usecols=["label", "message"],
        encoding="utf-8"
    )
    # Padroniza rótulos (caso seja necessário)
    df_phish["label"] = df_phish["label"].map({
        "ham": "ham",
        "spam": "spam",
        "smishing": "smishing"
    })
    return df_spam, df_phish

def inspect_distribution(df, name):
    print(f"=== Distribuição de classes em {name} ===")
    print(df["label"].value_counts(dropna=False))
    print()

def preprocess_and_vectorize(df_list, output_dir):
    # concatena para criar vocabulário global
    df_all = pd.concat(df_list, ignore_index=True)

    # limpeza básica
    df_all = df_all.drop_duplicates(subset="message")
    df_all = df_all.dropna(subset=["message", "label"])

    # pipeline: limpeza + tfidf
    pipe = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            preprocessor=None,       # já limpamos no TextCleaner
            tokenizer=None,          # usa o tokenizer interno
            min_df=5, max_df=0.9,
            ngram_range=(1,2)
        ))
    ])

    # treina o pipeline no texto completo
    X_tfidf = pipe.fit_transform(df_all["message"])

    # cria pasta common/ se não existir
    os.makedirs(output_dir, exist_ok=True)
    # salva o pipeline completo
    joblib.dump(pipe, os.path.join(output_dir, "preprocessing_pipeline.pkl"))
    print(f"Pipeline salvo em {output_dir}/preprocessing_pipeline.pkl")

    # (Opcional) salva apenas o vocabulário do TF-IDF
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    joblib.dump(tfidf.vocabulary_, os.path.join(output_dir, "tfidf_vocab.pkl"))
    print(f"Vocabulário TF-IDF salvo em {output_dir}/tfidf_vocab.pkl")

    return df_all, X_tfidf

if __name__ == "__main__":
    # caminhos para seus dados
    spam_path     = "data/raw/SMSSpamCollection.csv"
    phishing_path = "data/raw/SMS_Phishing_Dataset.csv"
    common_dir    = "src/common/"

    # 1) Carrega datasets
    df_spam, df_phish = load_datasets(spam_path, phishing_path)

    # 2) Inspeciona distribuição de classes
    inspect_distribution(df_spam,   "SMS Spam Collection")
    inspect_distribution(df_phish,  "SMS Phishing Dataset")

    # 3) Pré-processa, treina TF-IDF e salva pipeline/vocab
    df_all, X_tfidf = preprocess_and_vectorize(
        [df_spam, df_phish], common_dir
    )
