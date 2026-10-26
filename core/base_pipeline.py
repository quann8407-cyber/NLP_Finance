from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseFinancialPipeline(ABC):
    LANG = 'en'
    DEFAULT_INITIAL_CASH = 100000  # USD-scale, lớp con override cho VNĐ
    DEFAULT_LOT = 100

    def __init__(self, tickers, start, end):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.stock_data = None
        self.news_data = None
        self.lexicon = None
        self._chatbot = None

    @abstractmethod
    def load_stock_data(self): pass

    @abstractmethod
    def load_news_data(self): pass

    @abstractmethod
    def preprocess_text(self, text: str) -> str: pass

    @abstractmethod
    def load_lexicon(self) -> dict: pass

    def compute_sentiment_score(self, text: str) -> float:
        if self.lexicon is None:
            self.lexicon = self.load_lexicon()
        tokens = self.preprocess_text(text).split()
        return sum(self.lexicon.get(t, 0.0) for t in tokens)

    def run_case_study_1(self, signal_mode="sign",
                          quantile_pct=0.25, fixed_threshold=0.5,
                          liquidate_at_end=True,
                          initial_cash=None, lot=None):
        print(">> [Case Study 1] Sentiment Analysis Trading Strategy")
        if self.stock_data is None:
            self.load_stock_data()
        if self.news_data is None:
            self.load_news_data()

        if self.news_data.empty or 'Close' not in self.news_data.columns:
            print("[WARN] Thiếu dữ liệu giá (Close) để backtest.")
            return self.news_data

        # FIX: Dùng vốn/lot mặc định theo từng pipeline (VNĐ cần vốn lớn hơn
        # USD rất nhiều do thang giá khác nhau - 1 lot = 100 cổ phiếu VN giá
        # ~10,000-130,000 VNĐ cần 1-13 triệu VNĐ/lệnh, trong khi 100,000
        # "tiền" chỉ đủ cho thang giá USD của cổ phiếu Mỹ).
        initial_cash = initial_cash if initial_cash is not None else self.DEFAULT_INITIAL_CASH
        lot = lot if lot is not None else self.DEFAULT_LOT
        print(f"[INFO] Vốn ban đầu mỗi mã: {initial_cash:,.0f}, lot size: {lot}")

        df = self.news_data.copy()
        df['sentiment'] = df['headline'].apply(self.compute_sentiment_score)

        n_zero = (df['sentiment'] == 0).sum()
        n_pos = (df['sentiment'] > 0).sum()
        n_neg = (df['sentiment'] < 0).sum()
        print(f"[DEBUG] Phân bố sentiment: min={df['sentiment'].min():.3f}, "
              f"max={df['sentiment'].max():.3f}, mean={df['sentiment'].mean():.3f}")
        print(f"[DEBUG] Số dòng: dương={n_pos}, âm={n_neg}, bằng 0={n_zero}/{len(df)}")

        if signal_mode == "sign":
            df['signal'] = df['sentiment'].apply(
                lambda s: 'BUY' if s > 0 else ('SELL' if s < 0 else 'HOLD')
            )
        elif signal_mode == "quantile":
            nonzero = df[df['sentiment'] != 0]['sentiment']
            if len(nonzero) < 2 or nonzero.nunique() < 2:
                df['signal'] = df['sentiment'].apply(
                    lambda s: 'BUY' if s > 0 else ('SELL' if s < 0 else 'HOLD')
                )
            else:
                upper = nonzero.quantile(1 - quantile_pct)
                lower = nonzero.quantile(quantile_pct)
                df['signal'] = df['sentiment'].apply(
                    lambda s: 'BUY' if s >= upper else ('SELL' if s <= lower else 'HOLD')
                )
        else:
            df['signal'] = df['sentiment'].apply(
                lambda s: 'BUY' if s > fixed_threshold
                else ('SELL' if s < -fixed_threshold else 'HOLD')
            )

        n_buy = (df['signal'] == 'BUY').sum()
        n_sell = (df['signal'] == 'SELL').sum()
        print(f"[INFO] Số tín hiệu: BUY={n_buy}, SELL={n_sell}, "
              f"HOLD={len(df) - n_buy - n_sell}")

        if 'ticker' in df.columns:
            print("\n[DIAGNOSTIC] Chi tiết theo từng ticker (kèm giá min-max để "
                  "xác nhận biến động giá):")
            print(f"{'Ticker':<8}{'Tổng dòng':<11}{'Có giá':<9}"
                  f"{'BUY':<6}{'SELL':<6}{'Giá min':<12}{'Giá max':<12}"
                  f"{'1 lot cần':<14}")
            for ticker in sorted(df['ticker'].unique()):
                sub = df[df['ticker'] == ticker]
                n_total = len(sub)
                n_has_price = sub['Close'].notna().sum()
                n_buy_t = (sub['signal'] == 'BUY').sum()
                n_sell_t = (sub['signal'] == 'SELL').sum()
                prices = sub['Close'].dropna()
                pmin = prices.min() if len(prices) else 0
                pmax = prices.max() if len(prices) else 0
                cost_per_lot = pmin * lot
                afford_flag = "✓" if initial_cash >= cost_per_lot else "✗ KHÔNG ĐỦ VỐN"
                print(f"{ticker:<8}{n_total:<11}{n_has_price:<9}"
                      f"{n_buy_t:<6}{n_sell_t:<6}{pmin:<12,.0f}{pmax:<12,.0f}"
                      f"{cost_per_lot:<14,.0f}{afford_flag}")

        if 'ticker' in df.columns:
            results_per_ticker = {}
            trade_summary = {}
            all_results = []
            all_trade_logs = {}
            for ticker in df['ticker'].unique():
                sub = df[df['ticker'] == ticker].copy()
                if sub.empty:
                    continue
                sub_result, summary, trade_log = self._backtest_simple(
                    sub, initial_cash=initial_cash, lot=lot,
                    liquidate_at_end=liquidate_at_end
                )
                all_results.append(sub_result)
                final_value = sub_result['portfolio_value'].iloc[-1]
                results_per_ticker[ticker] = final_value
                trade_summary[ticker] = summary
                all_trade_logs[ticker] = trade_log

            result = pd.concat(all_results, ignore_index=True) if all_results else df

            # ---- In log giao dịch chi tiết theo yêu cầu: xem giá tại mỗi lần mua/bán ----
            print("\n[TRADE LOG] Chi tiết từng lệnh đã thực thi (để xác nhận biến động giá):")
            any_trade_executed = False
            for ticker, log in all_trade_logs.items():
                if log:
                    any_trade_executed = True
                    print(f"\n  {ticker}:")
                    for entry in log:
                        print(f"    {entry['action']:<5} @ {entry['price']:>12,.0f} VNĐ/USD "
                              f"(dòng {entry['row_idx']})")
            if not any_trade_executed:
                print("  (Không có lệnh nào được thực thi - kiểm tra bảng DIAGNOSTIC "
                      "phía trên cột 'Giá min' vs '1 lot cần' vs vốn ban đầu)")

            print("\n[KẾT QUẢ LÃI/LỖ] (đã thanh lý toàn bộ vị thế cuối kỳ):")
            print(f"{'Ticker':<8}{'Vốn đầu':<16}{'Kết thúc':<16}"
                  f"{'Lãi/Lỗ':<16}{'% Lãi/Lỗ':<10}{'Số lệnh':<10}")
            for ticker, final_value in results_per_ticker.items():
                summary = trade_summary.get(ticker, {})
                ic = summary.get('initial_cash', initial_cash)
                pnl = final_value - ic
                pnl_pct = (pnl / ic) * 100 if ic else 0
                n_trades = len(all_trade_logs.get(ticker, []))
                status = "LÃI" if pnl > 0 else ("LỖ" if pnl < 0 else "HÒA VỐN")
                print(f"{ticker:<8}{ic:<16,.0f}{final_value:<16,.0f}"
                      f"{pnl:<+16,.0f}{pnl_pct:<+9.2f}%{n_trades:<10}[{status}]")

            self.news_data = result
            return result, results_per_ticker
        else:
            result, _, _ = self._backtest_simple(
                df, initial_cash=initial_cash, lot=lot, liquidate_at_end=liquidate_at_end
            )
            self.news_data = result
            return result

    def _backtest_simple(self, df, initial_cash=100000, lot=100, liquidate_at_end=True):
        """
        Backtest đơn giản với BUY/SELL theo tín hiệu sentiment.
        Trả về (df_với_portfolio_value, summary_dict, trade_log_list)
        trade_log_list: list các dict {action, price, row_idx} cho MỌI lệnh
        THỰC SỰ được thực thi (không phải mọi tín hiệu, vì có thể bị chặn
        do thiếu vốn/thiếu vị thế).
        """
        cash, position, portfolio_values = initial_cash, 0, []
        last_valid_price = None
        trade_log = []

        for idx, row in df.iterrows():
            price = row.get('Close', np.nan)
            if pd.isna(price):
                portfolio_values.append(cash + position * (last_valid_price or 0))
                continue

            last_valid_price = price
            if row['signal'] == 'BUY' and cash >= price * lot:
                cash -= price * lot
                position += lot
                trade_log.append({'action': 'BUY', 'price': price, 'row_idx': idx})
            elif row['signal'] == 'SELL' and position >= lot:
                cash += price * lot
                position -= lot
                trade_log.append({'action': 'SELL', 'price': price, 'row_idx': idx})
            portfolio_values.append(cash + position * price)

        remaining_position_before_liquidation = position

        if liquidate_at_end and position > 0 and last_valid_price is not None:
            cash += position * last_valid_price
            trade_log.append({'action': 'LIQUIDATE', 'price': last_valid_price, 'row_idx': 'end'})
            position = 0
            if portfolio_values:
                portfolio_values[-1] = cash

        df = df.copy()
        df['portfolio_value'] = portfolio_values

        summary = {
            'initial_cash': initial_cash,
            'final_cash': cash,
            'remaining_position_before_liquidation': remaining_position_before_liquidation,
            'liquidated': liquidate_at_end and remaining_position_before_liquidation > 0,
        }
        return df, summary, trade_log

    def run_case_study_2(self, user_input: str = None, use_trained_ner=False):
        print(">> [Case Study 2] Chatbot Financial Ratio Assistant")
        from models.chatbot import FinancialChatbot
        if self._chatbot is None:
            self._chatbot = FinancialChatbot(lang=self.LANG, use_trained_ner=use_trained_ner)
        if user_input is None:
            user_input = ("What is the ROE for Apple?" if self.LANG == 'en'
                          else "Cho tôi xem tỷ số ROE của Vinamilk")
        result = self._chatbot.get_response(user_input)
        print(f"  Input : {user_input}")
        print(f"  Output: {result}")
        return result

    def run_case_study_3(self, documents: list, n_topics=5, n_top_words=10, visualize=False):
        print(">> [Case Study 3] Document Topic Modeling (LDA)")
        from models.lda_topic_model import run_lda_pipeline
        stop_words = 'english' if self.LANG == 'en' else None
        model, topics, panel = run_lda_pipeline(
            documents, n_topics=n_topics, n_words=n_top_words,
            visualize=visualize, stop_words=stop_words
        )
        return model, topics, panel
