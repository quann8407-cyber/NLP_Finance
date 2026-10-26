"""
LSTM-based sentiment classification model.
Bám theo Case Study 1, mục 4.2 trong sách.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense


class LSTMSentimentModel:
    def __init__(self, vocab_size=20000, max_len=50, embedding_dim=300):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.model = None
        self.history = None

    def prepare_sequences(self, texts, fit_tokenizer=True):
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, headlines, labels, epochs=3, validation_split=0.4, batch_size=32, verbose=1):
        X = self.prepare_sequences(headlines, fit_tokenizer=True)
        y = np.array(labels)
        if self.model is None:
            self.build_model()
        self.history = self.model.fit(X, y, epochs=epochs, validation_split=validation_split,
                                       batch_size=batch_size, verbose=verbose)
        return self.history

    def predict(self, texts):
        X = self.prepare_sequences(texts, fit_tokenizer=False)
        probs = self.model.predict(X, verbose=0)
        return (probs > 0.5).astype(int).flatten(), probs.flatten()

    def evaluate(self, texts, true_labels):
        preds, probs = self.predict(texts)
        acc = accuracy_score(true_labels, preds)
        print(f"[LSTM] Accuracy: {acc:.4f}")
        print(classification_report(true_labels, preds))
        return acc

    def save(self, path="models/saved/lstm_sentiment.h5"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[INFO] Đã lưu model LSTM tại {path}")

    def load(self, path="models/saved/lstm_sentiment.h5"):
        self.model = load_model(path)
        print(f"[INFO] Đã load model LSTM từ {path}")


def run_lstm_pipeline(csv_path="data/raw/LabelledNewsData.csv",
                       text_col="headline", label_col="sentiment",
                       test_size=0.2, epochs=3):
    print(">> [LSTM] Đang load dữ liệu đã gán nhãn...")
    try:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"[WARN] Không tìm thấy {csv_path}. Dùng dữ liệu demo.")
        df = pd.DataFrame({
            'headline': [
                "Apple posts record profit growth this quarter",
                "Company faces bankruptcy after massive losses",
                "Stock jumps after positive earnings beat",
                "Shares plunge amid fraud investigation"
            ],
            'sentiment': [1, 0, 1, 0]
        })

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=test_size, random_state=42
    )
    lstm = LSTMSentimentModel()
    lstm.train(X_train.tolist(), y_train.tolist(), epochs=epochs)
    lstm.evaluate(X_test.tolist(), y_test.tolist())
    lstm.save()
    return lstm


if __name__ == "__main__":
    run_lstm_pipeline()
