"""
Chatbot kết hợp:
- ChatterBot corpus (english.greetings, english.conversations) cho tiếng Anh
- List training thủ công cho tiếng Việt
- Custom logic adapter (FinancialRatioAdapter) để trả lời câu hỏi tỷ số tài chính

Đã fix lỗi: pip package `chatterbot_corpus` thường thiếu file dữ liệu YAML
thực tế (bug đóng gói đã biết). Giải pháp: tự động clone dữ liệu gốc từ
GitHub (gunthercox/chatterbot-corpus) và trỏ trainer vào path cục bộ.
"""

import os
import subprocess
import glob

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

CORPUS_CLONE_DIR = "external_corpus/chatterbot-corpus"


def _ensure_corpus_data():
    """
    Kiểm tra xem dữ liệu corpus (file .yml) có tồn tại chưa.
    Nếu chưa -> clone trực tiếp từ GitHub về thư mục cục bộ,
    trả về đường dẫn tới folder data/english để dùng trực tiếp
    (ChatterBotCorpusTrainer chấp nhận cả dotted-path lẫn path
    file hệ thống thật).
    """
    english_dir = os.path.join(CORPUS_CLONE_DIR, "chatterbot_corpus", "data", "english")

    if os.path.exists(english_dir) and glob.glob(os.path.join(english_dir, "*.yml")):
        return english_dir

    print("[INFO] Không tìm thấy dữ liệu corpus (.yml) -> đang clone từ GitHub...")
    os.makedirs("external_corpus", exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/gunthercox/chatterbot-corpus.git",
             CORPUS_CLONE_DIR],
            check=True, capture_output=True, text=True
        )
        print("[INFO] Clone corpus thành công.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Clone corpus thất bại: {e.stderr}")
        return None

    if os.path.exists(english_dir) and glob.glob(os.path.join(english_dir, "*.yml")):
        return english_dir
    print("[ERROR] Vẫn không tìm thấy dữ liệu corpus sau khi clone.")
    return None


def build_english_chatbot(name="TraderBot"):
    """Chatbot tiếng Anh dùng corpus có sẵn của ChatterBot (giống sách mục 3)"""
    chatbot = ChatBot(
        name,
        preprocessors=['chatterbot.preprocessors.clean_whitespace'],
        logic_adapters=[
            'chatterbot.logic.BestMatch',
            'chatterbot.logic.MathematicalEvaluation',
        ]
    )

    corpus_trainer = ChatterBotCorpusTrainer(chatbot)

    # ---- Thử cách chuẩn trước (dotted path qua pip package) ----
    trained_ok = False
    try:
        corpus_trainer.train("chatterbot.corpus.english")
        corpus_trainer.train("chatterbot.corpus.english.greetings")
        corpus_trainer.train("chatterbot.corpus.english.conversations")
        trained_ok = True
        print("[INFO] Train corpus thành công qua pip package chatterbot_corpus.")
    except (FileNotFoundError, OSError) as e:
        print(f"[WARN] pip package chatterbot_corpus thiếu dữ liệu ({e}). "
              f"Chuyển sang clone corpus từ GitHub...")

    # ---- Fallback: clone corpus từ GitHub, train bằng path cục bộ ----
    if not trained_ok:
        english_dir = _ensure_corpus_data()
        if english_dir:
            # Train toàn bộ folder english (bao gồm greetings, conversations, ai.yml,...)
            corpus_trainer.train(english_dir)
            print(f"[INFO] Train corpus thành công từ path cục bộ: {english_dir}")
            trained_ok = True
        else:
            print("[WARN] Không thể train corpus (thiếu dữ liệu). "
                  "Chatbot sẽ chỉ dùng list training + custom adapter.")

    list_trainer = ListTrainer(chatbot)
    list_trainer.train(['Help!', 'Please go to google.com'])
    list_trainer.train(['What is Bitcoin?', 'It is a decentralized digital currency'])

    print(f"[INFO] Đã huấn luyện chatbot tiếng Anh '{name}'.")
    return chatbot


def build_vietnamese_chatbot(name="TraderBotVN"):
    """
    Chatbot tiếng Việt - ChatterBot không có corpus tiếng Việt dựng sẵn,
    nên tự tạo tập hội thoại cơ bản bằng ListTrainer.
    """
    chatbot = ChatBot(
        name,
        preprocessors=['chatterbot.preprocessors.clean_whitespace'],
        logic_adapters=[
            'chatterbot.logic.BestMatch',
            'chatterbot.logic.MathematicalEvaluation',
        ]
    )

    list_trainer = ListTrainer(chatbot)

    basic_conversations = [
        ['Xin chào', 'Chào bạn, tôi có thể giúp gì cho bạn?'],
        ['Bạn khỏe không', 'Tôi ổn, cảm ơn bạn đã hỏi thăm'],
        ['Cảm ơn', 'Không có gì, rất vui được giúp bạn'],
        ['Tạm biệt', 'Hẹn gặp lại bạn'],
        ['Bạn là ai', 'Tôi là trợ lý ảo hỗ trợ tra cứu thông tin tài chính'],
        ['Bitcoin là gì', 'Bitcoin là một loại tiền điện tử phi tập trung'],
    ]
    for pair in basic_conversations:
        list_trainer.train(pair)

    print(f"[INFO] Đã huấn luyện chatbot tiếng Việt '{name}' với hội thoại tự tạo.")
    return chatbot


class FinancialChatbot:
    def __init__(self, lang='en', use_trained_ner=False):
        from core.financial_ratio_adapter import FinancialRatioAdapter
        self.lang = lang
        self.ratio_adapter = FinancialRatioAdapter(lang=lang, use_trained_ner=use_trained_ner)

        if lang == 'en':
            self.chatbot = build_english_chatbot()
        else:
            self.chatbot = build_vietnamese_chatbot()

    def get_response(self, user_input: str) -> str:
        ratio_response = self.ratio_adapter.process(user_input)
        fail_msgs = ["Sorry! Could not figure out what the user wants",
                     "Xin lỗi! Không xác định được yêu cầu của bạn"]
        if ratio_response not in fail_msgs:
            return ratio_response

        response = self.chatbot.get_response(user_input)
        return str(response)


if __name__ == "__main__":
    bot_en = FinancialChatbot(lang='en')
    for q in ["Hi", "What is the ROE for Apple?", "What is 78964 plus 5970"]:
        print(f"  '{q}' -> {bot_en.get_response(q)}")
