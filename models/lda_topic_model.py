"""
LDA topic modeling wrapper.
Bám theo Case Study 3 "Document Summarization" trong sách.
Hỗ trợ cả tiếng Anh (stop_words='english') và tiếng Việt (stop_words=None,
vì văn bản tiếng Việt đã được xóa stopwords ở bước tiền xử lý).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDATopicModel:
    def __init__(self, n_topics=5, stop_words='english', ngram_range=(1, 1)):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.dtm = None
        self.feature_names = None

    def fit(self, documents: list):
        self.dtm = self.vectorizer.fit_transform(documents)
        self.lda.fit(self.dtm)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self

    def transform(self, documents: list = None):
        dtm = self.vectorizer.transform(documents) if documents is not None else self.dtm
        return self.lda.transform(dtm)

    def print_top_words(self, n_words=10):
        sorting = np.argsort(self.lda.components_)[:, ::-1]
        topics = {}
        for topic_idx in range(self.n_topics):
            top_words = self.feature_names[sorting[topic_idx][:n_words]]
            topics[f"Topic {topic_idx + 1}"] = list(top_words)
            print(f"Topic {topic_idx + 1}: {list(top_words)}")
        return topics

    def visualize(self):
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
            return pyLDAvis.sklearn.prepare(self.lda, self.dtm, self.vectorizer)
        except Exception as e:
            print(f"[WARN] Không thể tạo pyLDAvis: {e}")
            return None

    def assign_dominant_topic(self, documents: list):
        doc_topic_matrix = self.transform(documents)
        return np.argmax(doc_topic_matrix, axis=1) + 1


def run_lda_pipeline(documents: list, n_topics=5, n_words=10,
                      visualize=False, stop_words='english'):
    print(f">> [LDA] Đang huấn luyện mô hình với {n_topics} topics "
          f"(stop_words={stop_words})...")
    model = LDATopicModel(n_topics=n_topics, stop_words=stop_words)
    model.fit(documents)
    topics = model.print_top_words(n_words=n_words)
    panel = model.visualize() if visualize else None
    return model, topics, panel


if __name__ == "__main__":
    demo_docs = [
        "The stock price of google jumps on the earning data today",
        "Google plunge on China Data",
        "Apple reports record profit growth this quarter",
        "Amazon revenue increases due to strong holiday sales",
    ]
    run_lda_pipeline(demo_docs, n_topics=3, n_words=5)
