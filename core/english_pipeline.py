"""
Pipeline tiếng Anh - bám theo Case Study 1 (Chapter 10) của sách
"Machine Learning and Data Science Blueprints for Finance".
"""

import os
import json
import zipfile
import warnings
import numpy as np
import pandas as pd

import yfinance as yf
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from core.base_pipeline import BaseFinancialPipeline

warnings.filterwarnings("ignore")
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)


class EnglishBookPipeline(BaseFinancialPipeline):

    LANG = 'en'

    def __init__(self,
                 tickers=None,
                 start='2010-01-01',
                 end='2018-12-31',
                 headline_zip_path="data/raw/Raw Headline Data.zip",
                 stock_lex_path="data/lexicons/stock_lex.csv"):
        default_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB',
                            'WMT', 'JPM', 'TSLA', 'NFLX', 'ADBE']
        super().__init__(tickers or default_tickers, start, end)

        self.headline_zip_path = headline_zip_path
        self.stock_lex_path = stock_lex_path

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.lstm_model = None

    def load_stock_data(self):
        print(">> [2.2] Đang tải dữ liệu giá cổ phiếu từ Yahoo Finance...")
        df_ticker_return = pd.DataFrame()
        for ticker in self.tickers:
            try:
                ticker_yf = yf.Ticker(ticker)
                data_temp = ticker_yf.history(start=self.start, end=self.end)
                data_temp['ticker'] = ticker
                data_temp = data_temp.reset_index()
                df_ticker_return = pd.concat([df_ticker_return, data_temp], ignore_index=True)
            except Exception as e:
                print(f"[WARN] Lỗi tải {ticker}: {e}")

        self.stock_data = df_ticker_return
        os.makedirs("data/processed", exist_ok=True)
        self.stock_data.to_csv("data/processed/Step3.2_ReturnData.csv", index=False)
        print(f"[INFO] Đã tải xong {len(self.stock_data)} dòng giá cổ phiếu.")
        return self.stock_data

    def _json_parser(self, json_data):
        from lxml import etree
        from io import StringIO
        xml_data = json_data['content']
        try:
            tree = etree.parse(StringIO(xml_data), parser=etree.HTMLParser())
        except Exception:
            return [], []

        headlines = tree.xpath("//h4[contains(@class, 'media-heading')]/a/text()")
        main_tickers = list(map(lambda x: x.replace('/symbol/', ''),
                                 tree.xpath("//div[contains(@class, 'media-left')]//a/@href")))
        final_headlines = [''.join(f.xpath('.//text()')) for f in
                            tree.xpath("//div[contains(@class, 'media-body')]/ul/li[1]")]
        if len(final_headlines) == 0:
            final_headlines = [''.join(f.xpath('.//text()')) for f in
                                tree.xpath("//div[contains(@class, 'media-body')]")]
        final_headlines = [f.replace(h, '').split('\xa0')[0].strip()
                            for f, h in zip(final_headlines, headlines)]
        return main_tickers, final_headlines

    def load_news_data(self):
        print(">> [3.1] Đang xử lý dữ liệu tin tức (JSON headlines)...")
        rows = []
        try:
            z = zipfile.ZipFile(self.headline_zip_path, "r")
            for fname in z.namelist():
                try:
                    file_data = z.open(fname).read()
                    json_data = json.loads(file_data)
                    date_str = fname.split('/')[-1].replace('.json', '')
                    tickers_parsed, headlines_parsed = self._json_parser(json_data)
                    for t, h in zip(tickers_parsed, headlines_parsed):
                        rows.append({'ticker': t, 'headline': h, 'date': date_str})
                except Exception:
                    continue
            data_df_news = pd.DataFrame(rows)
        except FileNotFoundError:
            print(f"[WARN] Không tìm thấy {self.headline_zip_path}. Dùng dữ liệu mẫu demo.")
            data_df_news = pd.DataFrame({
                'ticker': ['AAPL', 'GOOG', 'AMZN', 'TSLA'],
                'headline': [
                    "Apple (AAPL) jumps after record iPhone sales beat estimates",
                    "Google (GOOG) plunge on antitrust investigation concerns",
                    "Amazon (AMZN) posts strong earnings, shares surge",
                    "Tesla (TSLA) falls after production delay announcement"
                ],
                'date': pd.date_range('2018-01-01', periods=4).astype(str)
            })

        if self.stock_data is None:
            self.load_stock_data()

        stock_df = self.stock_data.copy()
        stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')
        stock_df = stock_df.sort_values(['ticker', 'date'])
        stock_df['ret_curr'] = stock_df.groupby('ticker')['Close'].pct_change()
        stock_df['eventRet'] = (stock_df['ret_curr']
                                 + stock_df.groupby('ticker')['ret_curr'].shift(-1)
                                 + stock_df.groupby('ticker')['ret_curr'].shift(1))

        data_df_news['date'] = pd.to_datetime(data_df_news['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        combined = pd.merge(data_df_news, stock_df, how='left', on=['date', 'ticker'])
        combined = combined[combined['ticker'].isin(self.tickers)]
        combined = combined[['ticker', 'headline', 'date', 'eventRet', 'Close']].dropna(subset=['headline'])

        self.news_data = combined
        print(f"[INFO] Số dòng dữ liệu tin tức đã ghép với giá: {len(self.news_data)}")
        return self.news_data

    def preprocess_text(self, text: str) -> str:
        doc = self.nlp(str(text).lower())
        tokens = [t.text for t in doc if t.text not in self.stop_words and t.is_alpha]
        return " ".join(tokens)

    def load_lexicon(self) -> dict:
        print(">> [4.3] Đang xây dựng lexicon (VADER + stock_lex.csv)...")
        try:
            stock_lex = pd.read_csv(self.stock_lex_path)
            stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score']) / 2
            stock_lex = dict(zip(stock_lex['Item'], stock_lex['sentiment']))
            stock_lex = {k: v for k, v in stock_lex.items() if len(str(k).split(' ')) == 1}

            max_v, min_v = max(stock_lex.values()), min(stock_lex.values())
            stock_lex_scaled = {
                k: (v / max_v * 4 if v > 0 else v / min_v * -4)
                for k, v in stock_lex.items()
            }

            final_lex = {}
            final_lex.update(stock_lex_scaled)
            final_lex.update(self.sia.lexicon)
            self.sia.lexicon = final_lex
            self.lexicon = final_lex
            print(f"[INFO] Lexicon tổng hợp có {len(final_lex)} từ.")
        except FileNotFoundError:
            print(f"[WARN] Không tìm thấy {self.stock_lex_path}. Dùng VADER lexicon mặc định.")
            self.lexicon = self.sia.lexicon
        return self.lexicon

    def compute_sentiment_score(self, text: str) -> float:
        if self.lexicon is None:
            self.load_lexicon()
        return self.sia.polarity_scores(str(text))['compound']

    def compute_sentiment_textblob(self, text: str) -> float:
        return TextBlob(str(text)).sentiment.polarity

    def compute_sentiment_lstm(self, texts: list):
        """
        FIX: trả về XÁC SUẤT liên tục (probs) thay vì nhãn nhị phân (0/1).
        Lý do: nếu dùng nhãn 0/1 và model dự đoán toàn bộ 1 lớp duy nhất
        (rất dễ xảy ra khi model train trên tập dữ liệu có phân bố khác
        hẳn với dữ liệu áp dụng), cột đó có phương sai = 0, khiến
        pandas.corr() trả về NaN cho toàn bộ hàng/cột liên quan - đây là
        quy tắc toán học chuẩn (Pearson correlation chia cho std=0),
        không phải bug. Dùng xác suất liên tục giúp giữ được biến động
        nhỏ, tránh tình trạng NaN toàn phần.
        """
        from models.lstm_sentiment import LSTMSentimentModel
        if self.lstm_model is None:
            self.lstm_model = LSTMSentimentModel()
            try:
                self.lstm_model.load("models/saved/lstm_sentiment.h5")
            except Exception as e:
                print(f"[WARN] Chưa có LSTM model đã train ({e}).")
                return np.full(len(texts), np.nan)
        preds, probs = self.lstm_model.predict(texts)
        return probs  # dùng xác suất (0.0 - 1.0) thay vì nhãn 0/1

    def run_case_study_1(self, use_lstm=False, liquidate_at_end=True):
        print(">> [Case Study 1] NLP & Sentiment Analysis-Based Trading Strategy")
        if self.stock_data is None:
            self.load_stock_data()
        if self.news_data is None:
            self.load_news_data()
        if self.lexicon is None:
            self.load_lexicon()

        df = self.news_data.copy()
        df['sentiment_textblob'] = df['headline'].apply(self.compute_sentiment_textblob)
        df['sentiment_lex'] = df['headline'].apply(self.compute_sentiment_score)

        if use_lstm:
            lstm_scores = self.compute_sentiment_lstm(df['headline'].tolist())
            df['sentiment_lstm'] = lstm_scores

            # Kiểm tra phương sai trước khi tính correlation để tránh NaN gây khó hiểu
            lstm_std = df['sentiment_lstm'].std()
            if pd.isna(lstm_std) or lstm_std == 0:
                print(f"[WARN] LSTM sentiment có phương sai = 0 (model dự đoán gần như "
                      f"cùng 1 giá trị cho mọi headline). Tương quan sẽ là NaN - đây là "
                      f"hạn chế của model, không phải lỗi tính toán. Std hiện tại: {lstm_std}")
            else:
                print(f"[INFO] LSTM sentiment - std: {lstm_std:.4f}, "
                      f"min: {df['sentiment_lstm'].min():.4f}, "
                      f"max: {df['sentiment_lstm'].max():.4f}")

        print("\n[INFO] Tương quan sentiment vs eventRet:")
        cols = ['sentiment_textblob', 'sentiment_lex', 'eventRet']
        if use_lstm:
            cols.insert(2, 'sentiment_lstm')
        print(df[cols].corr())

        df = df.sort_values(['ticker', 'date'])
        df['prev_sentiment'] = df.groupby('ticker')['sentiment_lex'].shift(1)
        df['sentiment_change'] = df['sentiment_lex'] - df['prev_sentiment']

        df['signal'] = 'HOLD'
        df.loc[df['sentiment_change'] >= 0.5, 'signal'] = 'BUY'
        df.loc[df['sentiment_change'] <= -0.5, 'signal'] = 'SELL'

        results = {}
        trade_summary = {}
        all_results = []
        for ticker in self.tickers:
            sub = df[df['ticker'] == ticker].copy()
            if sub.empty:
                continue
            # FIX: _backtest_simple giờ trả về 3 giá trị (df, summary, trade_log)
            sub_result, summary, trade_log = self._backtest_simple(sub, liquidate_at_end=liquidate_at_end)
            all_results.append(sub_result)
            final_value = sub_result['portfolio_value'].iloc[-1] if len(sub_result) else np.nan
            results[ticker] = final_value
            trade_summary[ticker] = summary
            print(f"  {ticker}: Ending Portfolio Value = {final_value:.2f}")

        print("\n[KẾT QUẢ LÃI/LỖ] (đã thanh lý toàn bộ vị thế cuối kỳ):")
        print(f"{'Ticker':<8}{'Vốn đầu':<14}{'Kết thúc':<14}"
              f"{'Lãi/Lỗ':<14}{'% Lãi/Lỗ':<10}")
        for ticker, final_value in results.items():
            if pd.isna(final_value):
                continue
            summary = trade_summary.get(ticker, {})
            initial_cash = summary.get('initial_cash', 100000)
            pnl = final_value - initial_cash
            pnl_pct = (pnl / initial_cash) * 100
            status = "LÃI" if pnl > 0 else ("LỖ" if pnl < 0 else "HÒA VỐN")
            print(f"{ticker:<8}{initial_cash:<14,.0f}{final_value:<14,.0f}"
                  f"{pnl:<+14,.0f}{pnl_pct:<+9.2f}% [{status}]")

        result = pd.concat(all_results, ignore_index=True) if all_results else df
        self.news_data = result
        return result, results


if __name__ == "__main__":
    pipeline = EnglishBookPipeline()
    pipeline.run_case_study_1()
