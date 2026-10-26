"""
Visualizer utilities - dùng để hiển thị kết quả đẹp hơn trong console
và xuất ra hình ảnh (wordcloud, biểu đồ portfolio, bảng topic).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# ============================================================
# 0. Console formatting helpers - làm log/console đẹp hơn
# ============================================================
def print_header(title: str, char="=", width=70):
    """In tiêu đề section lớn, căn giữa"""
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_subheader(title: str, char="-", width=60):
    """In tiêu đề nhỏ hơn cho từng bước con"""
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_kv_table(data: dict, title: str = None, value_fmt="{:.2f}"):
    """In 1 dict dạng bảng key-value căn chỉnh đẹp (VD: portfolio theo ticker)"""
    if title:
        print(f"\n{title}")
    if not data:
        print("  (không có dữ liệu)")
        return
    max_key_len = max(len(str(k)) for k in data.keys())
    for k, v in data.items():
        try:
            v_str = value_fmt.format(v)
        except (ValueError, TypeError):
            v_str = str(v)
        print(f"  {str(k).ljust(max_key_len)} : {v_str}")


def print_topics_table(topics: dict):
    """In danh sách topic LDA dạng bảng dễ đọc thay vì list thô"""
    print("\n  Chủ đề phát hiện được:")
    for topic_name, words in topics.items():
        words_str = ", ".join(words)
        print(f"  - {topic_name}: {words_str}")


# ============================================================
# 1. WordCloud - trực quan hóa từ khóa xuất hiện nhiều trong văn bản
# ============================================================
def plot_wordcloud(text: str, save_path=None, title=None, show=True):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="black", max_words=2000,
                    stopwords=stopwords, width=1200, height=800)
    wc.generate(text)

    plt.figure(figsize=(14, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    if title:
        plt.title(title, fontsize=18, color="white", pad=15)
        plt.gcf().patch.set_facecolor('black')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=plt.gcf().get_facecolor(), bbox_inches='tight')
        print(f"[INFO] Đã lưu wordcloud tại: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 2. Biểu đồ danh mục đầu tư (portfolio value theo ticker)
# ============================================================
def plot_portfolio_by_ticker(results_per_ticker: dict, initial_cash=100000,
                              save_path=None, title="Ket qua giao dich theo ma co phieu",
                              show=True):
    """
    Vẽ biểu đồ cột so sánh Ending Portfolio Value giữa các ticker,
    tô màu xanh (lãi) / đỏ (lỗ) so với vốn ban đầu.
    """
    if not results_per_ticker:
        print("[WARN] Không có dữ liệu portfolio để vẽ.")
        return

    tickers = list(results_per_ticker.keys())
    values = list(results_per_ticker.values())
    colors = ['#2ecc71' if v >= initial_cash else '#e74c3c' for v in values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(tickers, values, color=colors, edgecolor='black', linewidth=0.5)
    plt.axhline(y=initial_cash, color='gray', linestyle='--', linewidth=1,
                label=f'Von ban dau ({initial_cash:,.0f})')

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Gia tri danh muc (Portfolio Value)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"[INFO] Đã lưu biểu đồ portfolio tại: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_sentiment_distribution(sentiments, save_path=None,
                                 title="Phan bo diem Sentiment", show=True):
    """Histogram phân bố điểm sentiment"""
    plt.figure(figsize=(10, 5))
    plt.hist(sentiments, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Nguong trung lap (0)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Diem Sentiment")
    plt.ylabel("So luong bai bao")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"[INFO] Đã lưu biểu đồ phân bố sentiment tại: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_heatmap(corr_df, save_path=None,
                              title="Tuong quan Sentiment vs Return", show=True):
    """Heatmap tương quan giữa các phương pháp sentiment và eventRet"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, cmap='RdYlGn', vmin=-1, vmax=1)

    n_cols = len(corr_df.columns)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_df.columns)

    for i in range(n_cols):
        for j in range(n_cols):
            val = corr_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='black', fontsize=10)

    plt.colorbar(im, ax=ax, label='He so tuong quan')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"[INFO] Đã lưu heatmap tương quan tại: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. Topic modeling visualization (pyLDAvis)
# ============================================================
def plot_topic_model(lda_model, feature_matrix, vectorizer):
    try:
        import pyLDAvis
        import pyLDAvis.sklearn
        panel = pyLDAvis.sklearn.prepare(lda_model, feature_matrix, vectorizer)
        pyLDAvis.display(panel)
        return panel
    except Exception as e:
        print(f"[WARN] Không thể vẽ pyLDAvis: {e}")
        return None
