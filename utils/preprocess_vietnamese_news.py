"""
Tiền xử lý dữ liệu tin tức tài chính tiếng Việt (crawl từ VnExpress).
"""

import os
import re
import pandas as pd
from underthesea import word_tokenize

RAW_PATH = "data/raw/vietnamese_rawdata.csv"
STOPWORDS_PATH = "data/lexicons/vietnamese-stopwords.txt"
OUTPUT_PATH = "data/processed/processed_news.csv"


def load_stopwords(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as f:
            raw_words = [line.strip() for line in f if line.strip()]
        stopwords_set = set(w.replace(" ", "_") for w in raw_words)
        stopwords_set.update(raw_words)
        print(f"[INFO] Đã load {len(raw_words)} stopwords từ {path}")
        return stopwords_set
    except FileNotFoundError:
        print(f"[WARN] Không tìm thấy {path}. Dùng tập stopwords rút gọn mặc định.")
        return {"là", "và", "của", "có", "được", "cho", "này", "đó", "các",
                "một", "những", "trong", "với", "để", "khi", "đã", "sẽ"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\xa0\x92\x93\x94\u200b]", " ", text)
    text = re.sub(
        r"[^0-9a-zàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềềểếễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ\s]",
        " ", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vietnamese(text: str) -> str:
    if not text:
        return ""
    try:
        return word_tokenize(text, format="text")
    except Exception as e:
        print(f"[WARN] Lỗi tokenize: {e}")
        return text


def remove_stopwords(tokenized_text: str, stopwords: set) -> str:
    tokens = tokenized_text.split()
    return " ".join([t for t in tokens if t not in stopwords])


def preprocess_pipeline(text: str, stopwords: set) -> str:
    cleaned = clean_text(text)
    tokenized = tokenize_vietnamese(cleaned)
    return remove_stopwords(tokenized, stopwords)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"[INFO] Đang đọc dữ liệu từ {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"[INFO] Số dòng dữ liệu gốc: {len(df)}")

    df = df.rename(columns={
        "Tiêu đề": "title", "Đường dẫn": "url", "Nội dung chi tiết": "content"
    })
    df = df.dropna(subset=["title", "content"])
    df = df[df["content"].str.strip() != ""]

    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"[INFO] Đã xóa {before - len(df)} dòng trùng lặp URL")

    stopwords = load_stopwords(STOPWORDS_PATH)

    print("[INFO] Đang tiền xử lý tiêu đề...")
    df["title_clean"] = df["title"].apply(clean_text)
    df["title_processed"] = df["title"].apply(lambda x: preprocess_pipeline(x, stopwords))

    print("[INFO] Đang tiền xử lý nội dung chi tiết...")
    df["content_clean"] = df["content"].apply(clean_text)
    df["content_processed"] = df["content"].apply(lambda x: preprocess_pipeline(x, stopwords))
    df["content_word_count"] = df["content_processed"].apply(lambda x: len(x.split()))

    output_cols = ["title", "url", "title_clean", "title_processed",
                   "content", "content_clean", "content_processed", "content_word_count"]
    df_out = df[output_cols]

    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã lưu {len(df_out)} dòng vào {OUTPUT_PATH}")
    return df_out


if __name__ == "__main__":
    main()
