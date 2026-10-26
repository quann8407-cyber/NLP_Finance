"""
Chạy đầy đủ 3 Case Study cho cả pipeline tiếng Anh và tiếng Việt.
Đã tích hợp visualizer + pdf_parser để output đẹp hơn:
- Wordcloud cho văn bản Case Study 3
- Biểu đồ portfolio value theo ticker cho Case Study 1
- Heatmap tương quan sentiment vs return
- Bảng topic LDA được format rõ ràng
Tất cả hình ảnh được lưu vào results/{lang}/...
"""

import os
import pandas as pd

from core.english_pipeline import EnglishBookPipeline
from core.vietnamese_pipeline import VietnameseRealDataPipeline
from utils.pdf_parser import convert_pdf_to_txt, clean_pdf_text, pdf_stats
from utils.visualizer import (
    print_header, print_subheader, print_kv_table, print_topics_table,
    plot_wordcloud, plot_portfolio_by_ticker,
    plot_sentiment_distribution, plot_correlation_heatmap
)

RESULTS_DIR = "results"


def _results_path(lang, filename):
    path = os.path.join(RESULTS_DIR, lang, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ================================================================
# ENGLISH PIPELINE
# ================================================================
def run_english_pipeline():
    print_header("ENGLISH PIPELINE")

    eng = EnglishBookPipeline(
        tickers=['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB',
                 'WMT', 'JPM', 'TSLA', 'NFLX', 'ADBE'],
        start='2010-01-01',
        end='2018-12-31',
        headline_zip_path="data/raw/Raw Headline Data.zip",
        stock_lex_path="data/lexicons/stock_lex.csv"
    )

    # ---- Case Study 1 ----
    print_subheader("Case Study 1: Sentiment Analysis Trading Strategy")
    try:
        use_lstm = os.path.exists("models/saved/lstm_sentiment.h5")
        if not use_lstm and os.path.exists("data/raw/LabelledNewsData.csv"):
            from models.lstm_sentiment import run_lstm_pipeline
            run_lstm_pipeline(csv_path="data/raw/LabelledNewsData.csv", epochs=3)
            use_lstm = True

        result, results_per_ticker = eng.run_case_study_1(use_lstm=use_lstm) \
            if hasattr(eng, 'run_case_study_1') else (None, {})

        print_kv_table(results_per_ticker, title="Kết quả cuối cùng theo mã cổ phiếu:",
                       value_fmt="{:,.2f}")

        plot_portfolio_by_ticker(
            results_per_ticker,
            title="EN - Ending Portfolio Value theo mã cổ phiếu",
            save_path=_results_path("english", "portfolio_by_ticker.png"),
            show=False
        )

        if result is not None and 'sentiment_lex' in result.columns:
            plot_sentiment_distribution(
                result['sentiment_lex'].dropna(),
                title="EN - Phân bố Sentiment (Lexicon-based)",
                save_path=_results_path("english", "sentiment_distribution.png"),
                show=False
            )
            corr_cols = [c for c in ['sentiment_textblob', 'sentiment_lex',
                                      'sentiment_lstm', 'eventRet'] if c in result.columns]
            corr_df = result[corr_cols].corr()
            plot_correlation_heatmap(
                corr_df, title="EN - Tương quan Sentiment vs eventRet",
                save_path=_results_path("english", "correlation_heatmap.png"),
                show=False
            )
    except Exception as e:
        print(f"[ERROR] EN Case Study 1 lỗi: {e}")

    # ---- Case Study 2 ----
    print_subheader("Case Study 2: Chatbot Financial Ratio Assistant")
    try:
        eng.run_case_study_2("What is the ROE for Apple?")
        eng.run_case_study_2("Tell me the PE for Delta?")
        eng.run_case_study_2("Hi, how are you?")
    except Exception as e:
        print(f"[ERROR] EN Case Study 2 lỗi: {e}")

    # ---- Case Study 3 ----
    print_subheader("Case Study 3: Document Summarization (LDA)")
    try:
        raw = convert_pdf_to_txt("data/raw/10K.pdf")
        docs = [d for d in clean_pdf_text(raw) if d.strip() != ""]

        print_kv_table(pdf_stats(docs), title="Thống kê văn bản PDF (10-K):")

        plot_wordcloud(
            " ".join(docs), title="EN - WordCloud báo cáo 10-K",
            save_path=_results_path("english", "wordcloud_10k.png"), show=False
        )

        model, topics, panel = eng.run_case_study_3(docs, n_topics=5, n_top_words=10)
        print_topics_table(topics)

    except FileNotFoundError:
        print("[WARN] Không tìm thấy data/raw/10K.pdf -> bỏ qua EN Case Study 3.")
    except Exception as e:
        print(f"[ERROR] EN Case Study 3 lỗi: {e}")

    return eng


# ================================================================
# VIETNAMESE PIPELINE
# ================================================================
def run_vietnamese_pipeline():
    print_header("VIETNAMESE PIPELINE")

    viet = VietnameseRealDataPipeline(
        tickers=['ACB', 'BID', 'CTG', 'FPT', 'GAS',
                 'HPG', 'MBB', 'NKG', 'REE', 'VCB',
                 'VNM', 'VIC'],   # thêm 2 mã bổ sung, code sẽ tự thử tải
        start='2020-01-01',
        end='2024-12-31',
        news_csv_path="data/processed/processed_news.csv",
        stopwords_path="data/lexicons/vietnamese-stopwords.txt",
        lexicon_path="data/lexicons/viet_lexicon.csv"
    )

    # ---- Case Study 1 ----
    print_subheader("Case Study 1: Sentiment Analysis Trading Strategy")
    try:
        output = viet.run_case_study_1()
        result, results_per_ticker = output if isinstance(output, tuple) else (output, {})

        print_kv_table(results_per_ticker, title="Kết quả cuối cùng theo mã cổ phiếu:",
                       value_fmt="{:,.2f}")
        if results_per_ticker:
            plot_portfolio_by_ticker(
                results_per_ticker, title="VN - Ending Portfolio Value theo ma co phieu",
                save_path=_results_path("vietnamese", "portfolio_by_ticker.png"), show=False
            )
        if result is not None and 'sentiment' in result.columns:
            plot_sentiment_distribution(
                result['sentiment'].dropna(),
                title="VN - Phan bo Sentiment (Lexicon tu xay)",
                save_path=_results_path("vietnamese", "sentiment_distribution.png"), show=False
            )
    except Exception as e:
        print(f"[ERROR] VN Case Study 1 lỗi: {e}")

    # ---- Case Study 2 ----
    print_subheader("Case Study 2: Chatbot tra cứu tỷ số tài chính")
    try:
        viet.run_case_study_2("Cho tôi xem tỷ số ROE của Vinamilk")
        viet.run_case_study_2("Doanh thu của FPT là bao nhiêu")
        viet.run_case_study_2("Xin chào")
    except Exception as e:
        print(f"[ERROR] VN Case Study 2 lỗi: {e}")

    # ---- Case Study 3 ----
    print_subheader("Case Study 3: Tóm tắt tài liệu (LDA)")
    try:
        raw = convert_pdf_to_txt("data/raw/BaoCaoThuongNien.pdf")
        docs = [d for d in clean_pdf_text(raw) if d.strip() != ""]
        source_desc = "PDF Báo cáo thường niên"
    except FileNotFoundError:
        print("[WARN] Không tìm thấy data/raw/BaoCaoThuongNien.pdf.")
        print("       -> Vui lòng tải Annual Report PDF từ trang IR của công ty VN.")
        print("       -> Tạm thời dùng dữ liệu tin tức đã xử lý thay thế.")
        try:
            df = pd.read_csv("data/processed/processed_news.csv", encoding="utf-8-sig")
            docs = df['content'].dropna().tolist()
            source_desc = "Tin tức đã xử lý (fallback)"
        except Exception as e:
            print(f"[ERROR] VN Case Study 3 (fallback) lỗi: {e}")
            docs, source_desc = [], None
    except Exception as e:
        print(f"[ERROR] VN Case Study 3 lỗi: {e}")
        docs, source_desc = [], None

    if docs:
        try:
            print_kv_table(pdf_stats(docs), title=f"Thống kê văn bản ({source_desc}):")

            plot_wordcloud(
                " ".join(docs), title=f"VN - WordCloud ({source_desc})",
                save_path=_results_path("vietnamese", "wordcloud.png"), show=False
            )

            model, topics, panel = viet.run_case_study_3(docs, n_topics=5, n_top_words=10)
            print_topics_table(topics)
        except Exception as e:
            print(f"[ERROR] VN Case Study 3 (visualize) lỗi: {e}")

    return viet


if __name__ == "__main__":
    eng_pipeline = run_english_pipeline()
    viet_pipeline = run_vietnamese_pipeline()

    print_header("HOÀN TẤT CHẠY 3 CASE STUDY CHO CẢ 2 NGÔN NGỮ")
    print(f"📁 Tất cả biểu đồ/wordcloud đã được lưu vào thư mục: {RESULTS_DIR}/")
    print(f"   ├── english/   (portfolio, sentiment, correlation, wordcloud, ...)")
    print(f"   └── vietnamese/")
