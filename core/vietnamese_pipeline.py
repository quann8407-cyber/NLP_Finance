import os
import re
import numpy as np
import pandas as pd
from underthesea import word_tokenize

from core.base_pipeline import BaseFinancialPipeline

COMPANY_TO_TICKER = {
    'vinamilk': 'VNM', 'sữa việt nam': 'VNM',
    'fpt': 'FPT',
    'vietcombank': 'VCB', 'vcb': 'VCB', 'ngân hàng ngoại thương': 'VCB',
    'vietinbank': 'CTG', 'ctg': 'CTG', 'ngân hàng công thương': 'CTG',
    'bidv': 'BID', 'đầu tư và phát triển': 'BID',
    'ngân hàng đầu tư và phát triển': 'BID',
    'hòa phát': 'HPG', 'hoa phat': 'HPG', 'tập đoàn hòa phát': 'HPG',
    'acb': 'ACB', 'ngân hàng á châu': 'ACB',
    'mb bank': 'MBB', 'mbbank': 'MBB', 'quân đội': 'MBB',
    'ngân hàng quân đội': 'MBB',
    'nam kim': 'NKG', 'thép nam kim': 'NKG',
    'ree': 'REE', 'cơ điện lạnh': 'REE',
    'pv gas': 'GAS', 'khí việt nam': 'GAS', 'tổng công ty khí việt nam': 'GAS',
    'vingroup': 'VIC', 'vin group': 'VIC',
}

CSV_TICKERS = {'ACB', 'BID', 'CTG', 'FPT', 'GAS', 'HPG', 'MBB', 'NKG', 'REE', 'VCB'}
SUPPLEMENTARY_TICKERS = {'VNM', 'VIC'}


class VietnameseRealDataPipeline(BaseFinancialPipeline):

    LANG = 'vi'
    DEFAULT_INITIAL_CASH = 100_000_000  # 100 triệu VNĐ - đủ mua nhiều lot ở thang giá VN
    DEFAULT_LOT = 100

    def __init__(self, tickers, start, end,
                 news_csv_path="data/processed/processed_news.csv",
                 stopwords_path="data/lexicons/vietnamese-stopwords.txt",
                 lexicon_path="data/lexicons/viet_lexicon.csv",
                 price_csv_path="data/raw/data_10y.csv",
                 manual_csv_dir="data/raw/vn_prices",
                 try_supplementary=True):
        super().__init__(tickers, start, end)
        self.news_csv_path = news_csv_path
        self.stopwords_path = stopwords_path
        self.lexicon_path = lexicon_path
        self.price_csv_path = price_csv_path
        self.manual_csv_dir = manual_csv_dir
        self.try_supplementary = try_supplementary
        self.stop_words = self._load_stopwords()
        self._available_tickers = None

    def _load_stopwords(self):
        try:
            with open(self.stopwords_path, encoding='utf-8') as f:
                raw_words = [line.strip() for line in f if line.strip()]
            stopwords_set = set(w.replace(" ", "_") for w in raw_words)
            stopwords_set.update(raw_words)
            print(f"[INFO] Đã load {len(raw_words)} stopwords từ {self.stopwords_path}")
            return stopwords_set
        except Exception as e:
            print(f"[WARN] Không load được stopwords: {e}")
            return set()

    def _load_from_multiindex_csv(self, path):
        raw = pd.read_csv(path, header=[0, 1], index_col=0)
        raw.index.name = 'Date'
        raw = raw[raw.index.astype(str) != 'Date']
        raw.index = pd.to_datetime(raw.index, errors='coerce')
        raw = raw[raw.index.notna()]

        if raw.empty:
            return pd.DataFrame()

        tickers_in_csv = sorted(set(col[1] for col in raw.columns if col[1] != ''))

        frames = []
        for full_ticker in tickers_in_csv:
            try:
                sub = pd.DataFrame({
                    'Date': raw.index,
                    'Open': pd.to_numeric(raw[('Open', full_ticker)], errors='coerce').values,
                    'High': pd.to_numeric(raw[('High', full_ticker)], errors='coerce').values,
                    'Low': pd.to_numeric(raw[('Low', full_ticker)], errors='coerce').values,
                    'Close': pd.to_numeric(raw[('Close', full_ticker)], errors='coerce').values,
                    'Volume': pd.to_numeric(raw[('Volume', full_ticker)], errors='coerce').values,
                })
                clean_ticker = full_ticker.replace('.VN', '').replace('.HM', '').replace('.HN', '')
                sub['ticker'] = clean_ticker
                sub = sub.dropna(subset=['Close'])
                frames.append(sub)
            except KeyError as e:
                print(f"[WARN] Thiếu cột dữ liệu cho ticker {full_ticker}: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result[
            (result['Date'] >= pd.to_datetime(self.start)) &
            (result['Date'] <= pd.to_datetime(self.end))
        ]
        return result

    def _fetch_supplementary_prices(self, missing_tickers):
        if not missing_tickers:
            return pd.DataFrame()

        import yfinance as yf
        print(f"[INFO] Đang thử bổ sung giá cho các mã: {sorted(missing_tickers)} "
              f"(được nhắc nhiều trong tin tức nhưng không có trong CSV có sẵn)...")

        suffixes_to_try = ['.VN', '.HM', '.HN', '']
        frames = []
        for t in missing_tickers:
            fetched = False
            for suffix in suffixes_to_try:
                yf_ticker = f"{t}{suffix}"
                try:
                    df = yf.download(yf_ticker, start=self.start, end=self.end, progress=False)
                    if df is not None and not df.empty:
                        df = df.reset_index()
                        df['ticker'] = t
                        frames.append(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']])
                        print(f"  ✓ Tải thành công {yf_ticker}: {len(df)} dòng")
                        fetched = True
                        break
                except Exception:
                    continue
            if not fetched:
                print(f"  ✗ Không tải được giá cho {t}. Sẽ loại {t} khỏi ánh xạ ticker.")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_stock_data(self):
        base_data = pd.DataFrame()

        if os.path.exists(self.price_csv_path):
            print(f"[INFO] Đang đọc file giá có sẵn: {self.price_csv_path}")
            try:
                base_data = self._load_from_multiindex_csv(self.price_csv_path)
                if not base_data.empty:
                    print(f"[INFO] Đã đọc {len(base_data)} dòng, "
                          f"{base_data['ticker'].nunique()} mã: {sorted(base_data['ticker'].unique())}")
            except Exception as e:
                print(f"[WARN] Lỗi đọc {self.price_csv_path}: {e}")

        if self.try_supplementary:
            existing_tickers = set(base_data['ticker'].unique()) if not base_data.empty else set()
            missing = SUPPLEMENTARY_TICKERS - existing_tickers
            supp_data = self._fetch_supplementary_prices(missing)
            if not supp_data.empty:
                base_data = pd.concat([base_data, supp_data], ignore_index=True)

        if base_data.empty:
            print("[WARN] KHÔNG tải được giá cổ phiếu VN từ bất kỳ nguồn nào.")
            base_data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker'])

        self.stock_data = base_data
        self._available_tickers = set(base_data['ticker'].unique()) if not base_data.empty else set()
        return self.stock_data

    def _detect_ticker(self, title: str):
        title_lower = str(title).lower()
        for name, ticker in COMPANY_TO_TICKER.items():
            if name in title_lower:
                if self._available_tickers is None or ticker in self._available_tickers:
                    return ticker
                else:
                    return None
        return None

    def load_news_data(self):
        try:
            df = pd.read_csv(self.news_csv_path, encoding="utf-8-sig")
            df = df.rename(columns={"title": "headline"})
        except Exception as e:
            print(f"[WARN] load_news_data (VN) lỗi: {e}")
            df = pd.DataFrame({'headline': ["Vinamilk báo lãi tăng trưởng mạnh trong quý này"]})

        if self.stock_data is None:
            self.load_stock_data()

        n_total_headlines = len(df)
        df['ticker'] = df['headline'].apply(self._detect_ticker)

        df_all_mentions = df.copy()
        df_all_mentions['ticker_mentioned'] = df_all_mentions['headline'].apply(
            lambda t: next((tk for name, tk in COMPANY_TO_TICKER.items()
                            if name in str(t).lower()), None)
        )
        n_mentioned_no_price = ((df_all_mentions['ticker_mentioned'].notna()) &
                                 (df['ticker'].isna())).sum()
        if n_mentioned_no_price > 0:
            excluded_tickers = sorted(set(
                df_all_mentions.loc[
                    (df_all_mentions['ticker_mentioned'].notna()) & (df['ticker'].isna()),
                    'ticker_mentioned'
                ]
            ))
            print(f"[INFO] Đã loại {n_mentioned_no_price} bài báo nhắc tới các mã "
                  f"KHÔNG có giá khả dụng: {excluded_tickers}")

        df = df.dropna(subset=['ticker'])
        print(f"[INFO] Số bài báo gán được ticker (có giá hợp lệ): {len(df)}/{n_total_headlines}")

        if df.empty:
            self.news_data = df
            return self.news_data

        # ----------------------------------------------------------------
        # FIX QUAN TRỌNG: gán giá theo VỊ TRÍ THỜI GIAN (không phải trung
        # bình cố định) - mỗi bài báo của cùng 1 ticker sẽ lấy giá tại các
        # thời điểm khác nhau trải dài trong toàn bộ lịch sử giá của mã đó.
        # Điều này tạo ra biến động giá thực -> mua/bán ở các mức giá khác
        # nhau -> có thể LÃI hoặc LỖ thực sự, thay vì luôn hòa vốn tuyệt
        # đối như khi dùng giá trung bình cố định cho mọi giao dịch.
        # ----------------------------------------------------------------
        df['Close'] = np.nan
        if self.stock_data is not None and not self.stock_data.empty:
            for ticker in df['ticker'].unique():
                ticker_prices = self.stock_data[
                    self.stock_data['ticker'] == ticker
                ].sort_values('Date')['Close'].values

                idx = df[df['ticker'] == ticker].index
                n = len(idx)
                if len(ticker_prices) > 0 and n > 0:
                    sampled_positions = np.linspace(0, len(ticker_prices) - 1, n).astype(int)
                    df.loc[idx, 'Close'] = ticker_prices[sampled_positions]

        self.news_data = df
        n_valid = df['Close'].notna().sum()
        print(f"[INFO] Số dòng có giá Close hợp lệ: {n_valid}/{len(df)}")
        if n_valid > 0:
            print(f"[DEBUG] Khoảng giá theo ticker (min-max, để xác nhận giá đã biến động):")
            for ticker in df['ticker'].unique():
                prices = df[df['ticker'] == ticker]['Close'].dropna()
                if len(prices) > 0:
                    print(f"  {ticker}: min={prices.min():.2f}, max={prices.max():.2f}, "
                          f"số dòng={len(prices)}")
        return self.news_data

    def preprocess_text(self, text: str) -> str:
        tokens = word_tokenize(str(text).lower(), format="text").split()
        tokens = [t for t in tokens if t not in self.stop_words]
        return " ".join(tokens)

    def load_lexicon(self) -> dict:
        try:
            df = pd.read_csv(self.lexicon_path)
            self.lexicon = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            print(f"[INFO] Đã load lexicon VN với {len(self.lexicon)} từ.")
        except Exception as e:
            print(f"[WARN] load_lexicon (VN) lỗi: {e}")
            self.lexicon = {
                "tăng_trưởng": 1.0, "lợi_nhuận": 0.8,
                "thua_lỗ": -1.0, "phá_sản": -1.5
            }
        return self.lexicon

    def run_case_study_3(self, documents: list, n_topics=5, n_top_words=10, visualize=False):
        print(">> [Case Study 3 - VN] Document Topic Modeling (LDA) tiếng Việt")
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        processed_docs = [d for d in processed_docs if d.strip() != ""]
        return super().run_case_study_3(
            processed_docs, n_topics=n_topics, n_top_words=n_top_words,
            visualize=visualize
        )
