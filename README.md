# NLP_Finance

Dự án ứng dụng **xử lý ngôn ngữ tự nhiên (NLP) vào tài chính**, xây dựng pipeline end-to-end bằng Python cho **hai ngôn ngữ: tiếng Anh và tiếng Việt**. Các case study được xây dựng theo hướng dẫn trong tài liệu tham khảo (xem [Tài liệu tham khảo](#tài-liệu-tham-khảo)) và mở rộng sang dữ liệu thị trường chứng khoán Việt Nam.

Ba case study được triển khai cho cả hai ngôn ngữ:

| # | Case study | Nội dung |
|---|------------|----------|
| 1 | Sentiment Analysis Trading Strategy | Chấm điểm cảm xúc tiêu đề tin tức, sinh tín hiệu BUY/SELL/HOLD và backtest theo từng mã cổ phiếu |
| 2 | Chatbot Financial Ratio Assistant | Chatbot trả lời câu hỏi về tỷ số tài chính (ROE, P/E, doanh thu, dòng tiền hoạt động) |
| 3 | Document Summarization | Tóm tắt chủ đề tài liệu (báo cáo 10-K / báo cáo thường niên) bằng LDA và WordCloud |

---

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Chi tiết từng case study](#chi-tiết-từng-case-study)
- [Dữ liệu](#dữ-liệu)
- [Kết quả chạy thử](#kết-quả-chạy-thử)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Hạn chế đã biết](#hạn-chế-đã-biết)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Cấu trúc thư mục

```
NLP_Finance/
├── main.py                        # Chạy toàn bộ 3 case study cho EN và VI
├── requirements.txt
├── core/
│   ├── base_pipeline.py           # Lớp trừu tượng: sentiment, tín hiệu giao dịch, backtest
│   ├── english_pipeline.py        # EnglishBookPipeline (dữ liệu Yahoo Finance + headline Anh)
│   ├── vietnamese_pipeline.py     # VietnameseRealDataPipeline (tin VnExpress + giá cổ phiếu VN)
│   └── financial_ratio_adapter.py # Logic adapter cho chatbot (ánh xạ công ty / tỷ số, EN + VI)
├── models/
│   ├── lstm_sentiment.py          # Phân loại cảm xúc bằng LSTM (Keras)
│   ├── lda_topic_model.py         # Topic modeling LDA (scikit-learn)
│   ├── ner_model.py               # NER tùy biến (spaCy) cho nhãn COMPANY / RATIO
│   ├── chatbot.py                 # ChatterBot + FinancialRatioAdapter
│   └── saved/lstm_sentiment.h5    # Mô hình LSTM đã huấn luyện
├── utils/
│   ├── pdf_parser.py              # Đọc PDF (pdfminer.six), làm sạch văn bản, thống kê
│   ├── preprocess_vietnamese_news.py  # Tiền xử lý tin tức tiếng Việt
│   └── visualizer.py              # In bảng console, WordCloud, biểu đồ, heatmap
├── data/
│   ├── raw/                       # Dữ liệu gốc (giá, tin tức, PDF)
│   ├── processed/                 # Dữ liệu sau xử lý
│   └── lexicons/                  # Từ điển cảm xúc và stopwords
├── results/
│   ├── english/                   # Hình xuất ra của pipeline tiếng Anh
│   └── vietnamese/                # Hình xuất ra của pipeline tiếng Việt
├── caoData.ipynb                  # Notebook crawl tin tức tiếng Việt
├── writeFile.ipynb                # Notebook sinh mã nguồn dự án (Colab)
└── Test.ipynb                     # Notebook chạy main.py trên Google Colab
```

---

## Cài đặt

**Yêu cầu:** Python 3.12 (các file `.pyc` trong repo được biên dịch bằng 3.12), kết nối Internet để tải dữ liệu giá từ Yahoo Finance.

```bash
git clone https://github.com/quann8407-cyber/NLP_Finance.git
cd NLP_Finance

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install textblob             # được import trong english_pipeline.py nhưng chưa có trong requirements.txt
python -m spacy download en_core_web_sm
```

Ghi chú:

- Lần chạy đầu, `nltk` sẽ tự tải `stopwords` và `vader_lexicon`.
- Chatbot tiếng Anh cần dữ liệu YAML của `chatterbot-corpus`. Nếu thư mục `external_corpus/chatterbot-corpus` thiếu dữ liệu, `models/chatbot.py` sẽ tự `git clone` từ GitHub (cần cài sẵn `git`).
- `pyLDAvis.sklearn` có thể không khả dụng ở các phiên bản `pyLDAvis` mới. Khi đó phần trực quan hóa LDA được bỏ qua kèm cảnh báo, các phần còn lại vẫn chạy.

---

## Cách chạy

Chạy toàn bộ:

```bash
python main.py
```

`main.py` chạy lần lượt pipeline tiếng Anh rồi pipeline tiếng Việt, mỗi pipeline gồm Case Study 1, 2, 3. Mỗi case study được bọc trong `try/except`, nên một bước lỗi (ví dụ thiếu file dữ liệu) chỉ in `[ERROR]` hoặc `[WARN]` và không làm dừng toàn bộ chương trình.

Dùng riêng từng pipeline trong Python:

```python
from core.english_pipeline import EnglishBookPipeline
from core.vietnamese_pipeline import VietnameseRealDataPipeline

# Tiếng Anh
eng = EnglishBookPipeline(tickers=['AAPL', 'MSFT'], start='2010-01-01', end='2018-12-31')
result, per_ticker = eng.run_case_study_1(use_lstm=False)
eng.run_case_study_2("What is the ROE for Apple?")

# Tiếng Việt
viet = VietnameseRealDataPipeline(tickers=['FPT', 'VCB'], start='2020-01-01', end='2024-12-31')
result, per_ticker = viet.run_case_study_1()
viet.run_case_study_2("Cho tôi xem tỷ số ROE của Vinamilk")
```

Chạy trên Google Colab: mở `Test.ipynb`, gắn Google Drive và đặt dự án tại `MyDrive/NLP_Finance`.

---

## Chi tiết từng case study

### Case Study 1: Chiến lược giao dịch dựa trên sentiment

Luồng xử lý chung nằm ở `core/base_pipeline.py`:

1. Tải giá cổ phiếu và tin tức, ghép theo mã và ngày.
2. Tiền xử lý văn bản, tính điểm sentiment cho từng tiêu đề.
3. Sinh tín hiệu giao dịch theo một trong ba chế độ (`signal_mode`):
   - `"sign"`: điểm > 0 là BUY, < 0 là SELL, bằng 0 là HOLD
   - `"quantile"`: BUY/SELL theo phân vị trên/dưới của các điểm khác 0 (`quantile_pct`)
   - chế độ còn lại: ngưỡng cố định (`fixed_threshold`)
4. Backtest mỗi mã với vốn ban đầu và lot size cố định, có thể thanh lý toàn bộ vị thế cuối kỳ. In bảng chẩn đoán, log từng lệnh và bảng lãi/lỗ theo mã.

| | Tiếng Anh | Tiếng Việt |
|---|---|---|
| Mã cổ phiếu | AAPL, MSFT, AMZN, GOOG, FB, WMT, JPM, TSLA, NFLX, ADBE | ACB, BID, CTG, FPT, GAS, HPG, MBB, NKG, REE, VCB (+ VNM, VIC tải bổ sung) |
| Giai đoạn | 2010 – 2018 | 2020 – 2024 |
| Nguồn giá | Yahoo Finance (`yfinance`) | `data/raw/data_10y.csv`, bổ sung bằng `vnstock` hoặc CSV thủ công |
| Nguồn tin | `Raw Headline Data.zip` | Tin VnExpress đã crawl và tiền xử lý |
| Phương pháp sentiment | Lexicon `stock_lex.csv`, TextBlob, tùy chọn LSTM | Lexicon tự xây `viet_lexicon.csv` (khoảng 120 từ) |
| Tách từ | spaCy + stopwords NLTK | `underthesea` + stopwords tiếng Việt |
| Vốn / lot mặc định | 100,000 / 100 | 100,000,000 VNĐ / 100 |

Pipeline tiếng Anh tự huấn luyện LSTM (3 epoch) từ `data/raw/LabelledNewsData.csv` nếu chưa có `models/saved/lstm_sentiment.h5`.

### Case Study 2: Chatbot tra cứu tỷ số tài chính

- `core/financial_ratio_adapter.py` ánh xạ tên công ty và tên tỷ số (kể cả cách gọi khác nhau như "Vinamilk" và "Sữa Việt Nam") sang mã chuẩn, hỗ trợ `lang='en'` và `lang='vi'`.
- `models/ner_model.py` huấn luyện mô hình NER spaCy cho hai nhãn `COMPANY` và `RATIO` từ dữ liệu sinh theo template (có câu nhiễu và câu không chứa entity, tách train/dev theo tổ hợp công ty–tỷ số không trùng nhau, early stopping theo dev loss). Mặc định chatbot **không** dùng NER đã huấn luyện (`use_trained_ner=False`).
- `models/chatbot.py` kết hợp ChatterBot (corpus có sẵn cho tiếng Anh, danh sách huấn luyện thủ công cho tiếng Việt) với adapter ở trên.

Câu hỏi mẫu trong `main.py`: "What is the ROE for Apple?", "Tell me the PE for Delta?", "Cho tôi xem tỷ số ROE của Vinamilk", "Doanh thu của FPT là bao nhiêu".

### Case Study 3: Tóm tắt tài liệu bằng LDA

1. Đọc PDF bằng `pdfminer.six` (`data/raw/10K.pdf` cho tiếng Anh, `data/raw/BaoCaoThuongNien.pdf` cho tiếng Việt), làm sạch và tách dòng.
2. In thống kê (số dòng, tổng số từ, trung bình từ/dòng) và vẽ WordCloud.
3. Huấn luyện LDA (mặc định 5 chủ đề, 10 từ khóa mỗi chủ đề) và in bảng chủ đề.

Nếu thiếu `BaoCaoThuongNien.pdf`, pipeline tiếng Việt dùng tin tức đã xử lý (`processed_news.csv`) thay thế.

---

## Dữ liệu

| Đường dẫn | Mô tả |
|-----------|-------|
| `data/raw/data_10y.csv` | Giá lịch sử 10 mã cổ phiếu Việt Nam (OHLCV, định dạng MultiIndex của `yfinance`) |
| `data/raw/vietnamese_rawdata.csv` | Tin tức tài chính tiếng Việt crawl từ VnExpress (tiêu đề, đường dẫn, nội dung) |
| `data/raw/LabelledNewsData.csv` | Tiêu đề tin tiếng Anh có nhãn sentiment, dùng huấn luyện LSTM |
| `data/raw/Raw Headline Data.zip` | Dữ liệu headline thô cho pipeline tiếng Anh |
| `data/raw/10K.pdf`, `BaoCaoThuongNien.pdf` | Tài liệu cho Case Study 3 |
| `data/lexicons/` | `stock_lex.csv`, `viet_lexicon.csv`, `vietnamese-stopwords.txt` |
| `data/processed/` | Dữ liệu trung gian: `processed_news.csv`, các file `Step*.csv`, `Finance10k.txt` |

Để thu thập lại tin tức tiếng Việt, dùng `caoData.ipynb` (cần `beautifulsoup4`), sau đó chạy:

```bash
python -m utils.preprocess_vietnamese_news
```

Lệnh này đọc `data/raw/vietnamese_rawdata.csv` và ghi `data/processed/processed_news.csv`.

---

## Kết quả chạy thử

Số liệu dưới đây lấy từ log chạy `python main.py` trong `Test.ipynb` (Google Colab). Vốn ban đầu và lot size theo mặc định ở bảng phía trên; đã thanh lý toàn bộ vị thế cuối kỳ.

### Tiếng Anh (2010 – 2018, vốn 100,000 mỗi mã)

| Mã | Giá trị cuối kỳ | Lãi/Lỗ | % |
|----|----------------:|-------:|--:|
| AAPL | 111,614 | +11,614 | +11.61 |
| MSFT | 111,272 | +11,272 | +11.27 |
| AMZN | 118,493 | +18,493 | +18.49 |
| GOOG | 106,986 | +6,986 | +6.99 |
| FB | 100,000 | 0 | 0.00 |
| WMT | 101,417 | +1,417 | +1.42 |
| JPM | 101,725 | +1,725 | +1.73 |
| TSLA | 103,366 | +3,366 | +3.37 |
| NFLX | 109,994 | +9,994 | +9.99 |
| ADBE | 111,708 | +11,708 | +11.71 |

Tương quan giữa điểm sentiment và lợi suất sự kiện (`eventRet`): TextBlob 0.068, lexicon 0.158, LSTM 0.020. Cả ba đều thấp, trong đó lexicon (`stock_lex.csv` kết hợp VADER) cao nhất.

### Tiếng Việt (2020 – 2024, vốn 100,000,000 VNĐ mỗi mã)

| Mã | Giá trị cuối kỳ (VNĐ) | % | Số lệnh |
|----|----------------------:|--:|--------:|
| CTG | 119,644,941 | +19.64 | 20 |
| REE | 112,093,594 | +12.09 | 8 |
| VCB | 105,194,737 | +5.19 | 7 |
| ACB | 102,227,005 | +2.23 | 3 |
| BID | 101,933,230 | +1.93 | 4 |
| HPG | 101,357,499 | +1.36 | 2 |
| FPT | 100,000,000 | 0.00 | 2 |
| GAS | 100,000,000 | 0.00 | 0 |
| MBB | 100,000,000 | 0.00 | 0 |

### Chatbot

Câu hỏi tra tỷ số được trả về đường dẫn Zacks tương ứng, ví dụ `What is the ROE for Apple?` trả về `https://www.zacks.com/stock/chart/AAPL/fundamental/return-on-equity-ttm`. Chatbot không trả về giá trị số của tỷ số.

### LDA

Mỗi tài liệu được tách thành 5 chủ đề. Với báo cáo 10-K, các từ khóa nổi bật gồm `loans`, `credit`, `securities`, `capital`, `mortgage`, `fair value`. Với báo cáo thường niên tiếng Việt, các từ khóa gồm `chiến_lược`, `phát_triển`, `quản_trị`, `kinh_doanh`, `esg`. Nhiều chủ đề còn lẫn số và tháng (`2015`, `12`, `31`, `06`), nên chất lượng chủ đề còn hạn chế.

---

## Kết quả đầu ra

Hình được lưu tự động vào `results/<ngôn ngữ>/`:

- `portfolio_by_ticker.png`: giá trị danh mục cuối kỳ theo từng mã
- `sentiment_distribution.png`: phân bố điểm sentiment
- `correlation_heatmap.png`: tương quan giữa các thang sentiment và lợi suất sự kiện (chỉ có ở pipeline tiếng Anh)
- `wordcloud_10k.png` (EN), `wordcloud.png` (VI): WordCloud của tài liệu

---

## Hạn chế đã biết

- **Mã FB không có dữ liệu giá** trên Yahoo Finance trong lần chạy thử (báo "possibly delisted"), nên không có lệnh nào và kết quả là hòa vốn. Mã hiện hành của công ty là META.
- **Sentiment từ LSTM gần như không phân biệt được tin**: trong lần chạy thử, độ lệch chuẩn của điểm LSTM bằng 0 (mọi tiêu đề cùng nhận 0.8053), nên cột này không mang thông tin. Cần huấn luyện lại hoặc kiểm tra mô hình `lstm_sentiment.h5`.
- **Pipeline tiếng Việt chỉ gán được ticker cho 111/2,323 bài báo**; trong số đó 72/111 dòng có sentiment bằng 0 (HOLD), chỉ có 38 tín hiệu BUY và 1 tín hiệu SELL. Một số mã chỉ có 0 đến 3 dòng tin (GAS, MBB, HPG), nên lãi/lỗ theo mã dựa trên rất ít quan sát.
- **Chiến lược giao dịch chỉ mang tính minh họa học thuật.** Backtest đơn giản (lot cố định, không tính phí giao dịch, trượt giá, thuế), không có kiểm định ngoài mẫu. Kết quả không phải khuyến nghị đầu tư.
- **Lexicon tiếng Việt nhỏ** (khoảng 120 từ), nên nhiều tiêu đề có điểm sentiment bằng 0 và bị gán HOLD.
- **Dữ liệu chatbot tỷ số tài chính là bảng ánh xạ cố định** cho một số ít công ty và tỷ số, không phải truy vấn dữ liệu tài chính thời gian thực.
- **Phụ thuộc dữ liệu bên ngoài**: Yahoo Finance, `vnstock` và việc clone corpus từ GitHub có thể thay đổi hoặc lỗi theo thời gian.
- **Môi trường**: `requirements.txt` chưa ghim phiên bản và còn thiếu `textblob`; `pyLDAvis.sklearn` có thể không dùng được ở bản mới.
- **Repo còn file sinh ra khi chạy** (`__pycache__/`, `db.sqlite3` của ChatterBot); nên thêm `.gitignore` nếu muốn repo gọn hơn.

---

## Tài liệu tham khảo

- Hilpisch, Y. (2018). *Python for Finance: Mastering Data-Driven Finance* (2nd ed.). O'Reilly Media. Tài liệu nền tảng về Python, pandas và dữ liệu chuỗi thời gian tài chính (sách không có nội dung về sentiment, chatbot hay LDA).
- *Machine Learning and Data Science Blueprints for Finance*, Chapter 10 (NLP). Docstring trong `core/english_pipeline.py`, `models/lda_topic_model.py`, `models/lstm_sentiment.py` và `models/ner_model.py` ghi các case study bám theo sách này. Bạn kiểm tra lại tên tác giả và nhà xuất bản trước khi ghi vào README.
