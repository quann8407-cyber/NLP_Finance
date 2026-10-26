"""
Custom Named Entity Recognition (NER) model bằng spaCy.
Bám theo Case Study 2, mục 4-5 "Chatbot Digital Assistant" trong sách.

Đã sửa lỗi overfitting của phiên bản trước:
1. Mở rộng templates (đa dạng cấu trúc câu, có nhiễu, có câu không entity)
2. Sửa lỗi sent.index() gán sai vị trí khi entity là chuỗi con
3. Tách train/dev theo tổ hợp company-ratio KHÔNG trùng nhau (đo generalization thật)
4. Thêm early stopping dựa trên dev loss
5. Giới hạn/loại bỏ câu trùng lặp
"""

import random
import re
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example


class FinancialNERModel:
    def __init__(self, labels=None):
        self.labels = labels or ['RATIO', 'COMPANY']
        self.nlp = spacy.blank("en")
        self.ner = self.nlp.add_pipe("ner")
        for label in self.labels:
            self.ner.add_label(label)

    # ------------------------------------------------------------
    # 1. Templates đa dạng hơn nhiều (khác cấu trúc, có nhiễu, có câu
    #    không chứa entity nào để model học "không phải câu nào cũng
    #    có RATIO/COMPANY")
    # ------------------------------------------------------------
    DEFAULT_TEMPLATES = [
        'Get me the {ratio} for {company}',
        'What is the {ratio} for {company}?',
        'Tell me the {ratio} for {company}',
        'Can you show the {ratio} of {company}?',
        'I want to know the {ratio} for {company}',
        'Please provide {company} {ratio}',
        'How is {company} doing in terms of {ratio}?',
        'Show me {company}\'s {ratio} please',
        '{company} - what is its {ratio}?',
        'Give me {ratio} data for {company} today',
        'Looking for the {ratio} of {company}, can you help?',
        'What about the {ratio} for {company} this quarter?',
    ]

    # Câu KHÔNG chứa entity nào - giúp model không "ảo giác" entity ở mọi câu
    NEGATIVE_SENTENCES = [
        "Hello, how are you today?",
        "Can you help me with something?",
        "What is the weather like?",
        "Thank you very much for your help",
        "I would like to schedule a meeting",
        "This is a great conversation",
        "Please tell me a joke",
        "What time is it right now?",
    ]

    @staticmethod
    def _find_all_occurrences(text, sub):
        """Tìm TẤT CẢ vị trí xuất hiện của sub trong text (dùng word boundary
        để tránh match nhầm chuỗi con, ví dụ 'PE' không match trong 'Pepsi')."""
        pattern = r'\b' + re.escape(sub) + r'\b'
        return [m.start() for m in re.finditer(pattern, text)]

    @classmethod
    def generate_training_sample(cls, string_templates, ratios_list, companies_list):
        """
        Sinh 1 câu training, đã sửa lỗi gán sai vị trí entity khi
        ratio/company là chuỗi con của từ khác trong câu.
        """
        template = random.choice(string_templates)
        ratio = random.choice(ratios_list)
        company = random.choice(companies_list)
        sent = template.format(ratio=ratio, company=company)

        ratio_positions = cls._find_all_occurrences(sent, ratio)
        company_positions = cls._find_all_occurrences(sent, company)

        if not ratio_positions or not company_positions:
            # Không tìm được vị trí an toàn (word boundary) -> bỏ qua mẫu này
            return None

        # Lấy vị trí đầu tiên tìm được với word boundary (an toàn hơn .index())
        ratio_start = ratio_positions[0]
        company_start = company_positions[0]

        entities = [
            (ratio_start, ratio_start + len(ratio), 'RATIO'),
            (company_start, company_start + len(company), 'COMPANY'),
        ]
        # Sắp xếp entity theo vị trí xuất hiện (spaCy yêu cầu thứ tự tăng dần)
        entities.sort(key=lambda x: x[0])

        return (sent, {"entities": entities})

    @classmethod
    def generate_negative_sample(cls):
        """Sinh câu không chứa entity nào (annotation rỗng)"""
        sent = random.choice(cls.NEGATIVE_SENTENCES)
        return (sent, {"entities": []})

    def generate_training_data(self, string_templates, ratios_list, companies_list,
                                n_samples=100, negative_ratio=0.15, dedup=True):
        """
        Sinh dữ liệu training với:
        - negative_ratio: tỷ lệ câu không chứa entity (mặc định 15%)
        - dedup: loại bỏ câu trùng lặp hoàn toàn để tránh model học thuộc
        """
        data = []
        seen_sentences = set()
        n_negative = int(n_samples * negative_ratio)
        n_positive = n_samples - n_negative

        attempts = 0
        max_attempts = n_positive * 5  # tránh vòng lặp vô hạn nếu combo gần hết
        while len(data) < n_positive and attempts < max_attempts:
            attempts += 1
            sample = self.generate_training_sample(string_templates, ratios_list, companies_list)
            if sample is None:
                continue
            sent, ents = sample
            if dedup and sent in seen_sentences:
                continue
            seen_sentences.add(sent)
            data.append(sample)

        for _ in range(n_negative):
            data.append(self.generate_negative_sample())

        random.shuffle(data)
        print(f"[INFO] Đã sinh {len(data)} câu training "
              f"({n_positive} có entity, {n_negative} không có entity, "
              f"{len(seen_sentences)} câu duy nhất).")
        return data

    def generate_train_dev_split(self, string_templates, ratios_list, companies_list,
                                  n_train=150, n_dev=40, negative_ratio=0.15):
        """
        Tách train/dev theo tổ hợp company-ratio KHÔNG TRÙNG NHAU,
        để đo được model có generalize thật hay chỉ học thuộc.
        """
        all_combos = [(r, c) for r in ratios_list for c in companies_list]
        random.shuffle(all_combos)

        n_dev_combos = max(1, len(all_combos) // 5)  # ~20% combo dành cho dev
        dev_combos = set(all_combos[:n_dev_combos])
        train_combos = set(all_combos[n_dev_combos:])

        def _sample_from_combos(combos, n_samples, negative_ratio):
            data = []
            seen = set()
            n_negative = int(n_samples * negative_ratio)
            n_positive = n_samples - n_negative
            attempts, max_attempts = 0, n_positive * 8

            combos_list = list(combos)
            while len(data) < n_positive and attempts < max_attempts and combos_list:
                attempts += 1
                ratio, company = random.choice(combos_list)
                template = random.choice(string_templates)
                sent = template.format(ratio=ratio, company=company)

                ratio_pos = self._find_all_occurrences(sent, ratio)
                company_pos = self._find_all_occurrences(sent, company)
                if not ratio_pos or not company_pos:
                    continue
                if sent in seen:
                    continue
                seen.add(sent)

                ents = sorted([
                    (ratio_pos[0], ratio_pos[0] + len(ratio), 'RATIO'),
                    (company_pos[0], company_pos[0] + len(company), 'COMPANY'),
                ], key=lambda x: x[0])
                data.append((sent, {"entities": ents}))

            for _ in range(n_negative):
                data.append(self.generate_negative_sample())

            random.shuffle(data)
            return data

        train_data = _sample_from_combos(train_combos, n_train, negative_ratio)
        dev_data = _sample_from_combos(dev_combos, n_dev, negative_ratio)

        print(f"[INFO] Train: {len(train_data)} câu (từ {len(train_combos)} tổ hợp company-ratio)")
        print(f"[INFO] Dev  : {len(dev_data)} câu (từ {len(dev_combos)} tổ hợp company-ratio "
              f"KHÔNG xuất hiện trong train)")
        return train_data, dev_data

    # ------------------------------------------------------------
    # 2. Huấn luyện với early stopping dựa trên dev loss
    # ------------------------------------------------------------
    def train(self, train_data, dev_data=None, n_iter=30, drop=0.35,
              patience=5, min_delta=0.001):
        """
        patience: số vòng lặp cho phép dev loss không cải thiện trước khi dừng sớm
        min_delta: mức cải thiện tối thiểu để tính là "có tiến bộ"
        """
        optimizer = self.nlp.begin_training()
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]

        best_dev_loss = float('inf')
        patience_counter = 0
        best_model_bytes = None

        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(1.0, 4.0, 1.001)
            for itn in range(n_iter):
                random.shuffle(train_data)
                batches = minibatch(train_data, size=sizes)
                losses = {}
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        examples.append(Example.from_dict(doc, annotations))
                    self.nlp.update(examples, sgd=optimizer, drop=drop, losses=losses)

                train_loss = losses.get('ner', 0.0)

                if dev_data:
                    dev_loss = self._evaluate_loss(dev_data)
                    print(f"  Iter {itn + 1}/{n_iter} - Train loss: {train_loss:.4f} "
                          f"- Dev loss: {dev_loss:.4f}")

                    if dev_loss < best_dev_loss - min_delta:
                        best_dev_loss = dev_loss
                        patience_counter = 0
                        best_model_bytes = self.nlp.to_bytes()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"[INFO] Early stopping tại iter {itn + 1} "
                                  f"(dev loss không cải thiện sau {patience} vòng)")
                            break
                else:
                    if (itn + 1) % 5 == 0 or itn == n_iter - 1:
                        print(f"  Iter {itn + 1}/{n_iter} - Train loss: {train_loss:.4f}")

        # Khôi phục lại model tốt nhất theo dev loss (nếu có dùng dev_data)
        if best_model_bytes is not None:
            self.nlp.from_bytes(best_model_bytes)
            print(f"[INFO] Đã khôi phục model tốt nhất (dev loss = {best_dev_loss:.4f})")

        print("[INFO] Huấn luyện NER hoàn tất.")
        return self.nlp

    def _evaluate_loss(self, dev_data):
        """Tính loss trên tập dev mà KHÔNG cập nhật trọng số (để đánh giá generalization)"""
        examples = []
        for text, annotations in dev_data:
            doc = self.nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        losses = {}
        # sgd=None + drop=0.0 để chỉ đo loss, không update weights
        self.nlp.get_pipe("ner").update(examples, sgd=None, losses=losses)
        return losses.get('ner', 0.0)

    def evaluate_accuracy(self, dev_data):
        """Đánh giá độ chính xác nhận diện entity trên tập dev (đo bằng exact match)"""
        correct, total = 0, 0
        for text, annotations in dev_data:
            true_ents = set(tuple(e) for e in annotations['entities'])
            doc = self.nlp(text)
            pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
            if true_ents == pred_ents:
                correct += 1
            total += 1
        acc = correct / total if total > 0 else 0.0
        print(f"[INFO] Dev exact-match accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    # ------------------------------------------------------------
    # 3. Trích xuất entity từ câu mới
    # ------------------------------------------------------------
    def extract_entities(self, text: str) -> dict:
        doc = self.nlp(text)
        result = {'RATIO': None, 'COMPANY': None}
        for ent in doc.ents:
            if ent.label_ in result:
                result[ent.label_] = ent.text
        return result

    def save(self, path="models/saved/ner_model"):
        self.nlp.to_disk(path)
        print(f"[INFO] Đã lưu NER model tại {path}")

    def load(self, path="models/saved/ner_model"):
        self.nlp = spacy.load(path)
        print(f"[INFO] Đã load NER model từ {path}")


# ============================================================
# Hàm tiện ích chạy nhanh toàn bộ pipeline train + test
# ============================================================
def run_ner_pipeline(n_train=150, n_dev=40, n_iter=30, patience=5):
    ratios_list = ['ROE', 'Return on Equity', 'PE', 'pe ratio', 'Sales',
                   'Revenue', 'CFO', 'Cash Flow']
    companies_list = ['Apple', 'Apple Inc', 'Citi', 'Citibank',
                       'Delta', 'Delta Airlines', 'BofA', 'Bank of America',
                       'Google', 'Amazon', 'Tesla', 'Microsoft']

    ner_model = FinancialNERModel()

    print(">> [NER] Đang sinh dữ liệu train/dev (tách theo tổ hợp company-ratio)...")
    train_data, dev_data = ner_model.generate_train_dev_split(
        FinancialNERModel.DEFAULT_TEMPLATES, ratios_list, companies_list,
        n_train=n_train, n_dev=n_dev
    )

    print(f">> [NER] Đang huấn luyện (tối đa {n_iter} vòng, early stopping patience={patience})...")
    ner_model.train(train_data, dev_data=dev_data, n_iter=n_iter, patience=patience)

    print("\n>> [NER] Đánh giá trên tập dev (tổ hợp company-ratio CHƯA từng thấy khi train):")
    ner_model.evaluate_accuracy(dev_data)

    test_sentences = [
        "What is the ROE for Citibank?",
        "Tell me the PE for Delta?",
        "Get me the Revenue for Apple",
        "Hello, how are you doing today?",  # câu không có entity - test negative case
    ]
    print("\n=== Kiểm tra model trên câu mới ===")
    for s in test_sentences:
        print(f"  '{s}' -> {ner_model.extract_entities(s)}")

    ner_model.save()
    return ner_model


if __name__ == "__main__":
    run_ner_pipeline()
