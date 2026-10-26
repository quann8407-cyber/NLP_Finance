"""
Custom logic adapter cho Case Study 2 - hỗ trợ cả tiếng Anh và tiếng Việt.
"""

# ============================================================
# Dữ liệu ánh xạ - Tiếng Anh
# ============================================================
companies_en = {
    'AAPL': ['Apple', 'Apple Inc'],
    'GOOG': ['Google'],
    'C': ['Citi', 'Citibank'],
    'DAL': ['Delta', 'Delta Airlines'],
    'BAC': ['BAML', 'BofA', 'Bank of America'],
}

ratios_en = {
    'return-on-equity-ttm': ['ROE', 'Return on Equity'],
    'pe-ratio-ttm': ['PE', 'pe ratio', 'Price to equity'],
    'revenue-ttm': ['Sales', 'Revenue'],
    'cash-from-operations-quarterly': ['CFO', 'Cash Flow from Operations'],
}

# ============================================================
# Dữ liệu ánh xạ - Tiếng Việt
# ============================================================
companies_vi = {
    'VNM': ['Vinamilk', 'Sữa Việt Nam'],
    'FPT': ['FPT'],
    'VCB': ['Vietcombank', 'VCB'],
    'HPG': ['Hòa Phát', 'Hoa Phat'],
    'VIC': ['Vingroup', 'VinGroup'],
}

ratios_vi = {
    'return-on-equity-ttm': ['ROE', 'Tỷ suất lợi nhuận trên vốn chủ sở hữu'],
    'pe-ratio-ttm': ['PE', 'Tỷ số giá trên thu nhập', 'Hệ số P/E'],
    'revenue-ttm': ['Doanh thu', 'Doanh số'],
    'cash-from-operations-quarterly': ['Dòng tiền hoạt động', 'CFO'],
}


class FinancialRatioAdapter:
    """
    Adapter dùng chung cho cả 2 ngôn ngữ.
    lang='en' -> dùng companies_en/ratios_en
    lang='vi' -> dùng companies_vi/ratios_vi
    """

    def __init__(self, lang='en', use_trained_ner=False,
                 ner_model_path="models/saved/ner_model"):
        self.lang = lang
        if lang == 'vi':
            companies, ratios = companies_vi, ratios_vi
        else:
            companies, ratios = companies_en, ratios_en

        self.companies_rev = {v: k for k, vs in companies.items() for v in vs}
        self.ratios_rev = {v: k for k, vs in ratios.items() for v in vs}

        self.use_trained_ner = use_trained_ner
        self.ner_model = None
        if use_trained_ner:
            try:
                from models.ner_model import FinancialNERModel
                self.ner_model = FinancialNERModel()
                self.ner_model.load(ner_model_path)
            except Exception as e:
                print(f"[WARN] Không load được NER model đã train ({e}). "
                      f"Chuyển sang keyword-matching.")
                self.use_trained_ner = False

    def process(self, user_input: str) -> str:
        found_company, found_ratio = None, None

        if self.use_trained_ner and self.ner_model is not None:
            entities = self.ner_model.extract_entities(user_input)
            found_ratio = entities.get('RATIO')
            found_company = entities.get('COMPANY')
        else:
            text_lower = user_input.lower()
            for name in self.companies_rev:
                if name.lower() in text_lower:
                    found_company = name
                    break
            for name in self.ratios_rev:
                if name.lower() in text_lower:
                    found_ratio = name
                    break

        if found_company in self.companies_rev and found_ratio in self.ratios_rev:
            ticker = self.companies_rev[found_company]
            ratio_code = self.ratios_rev[found_ratio]
            return f"https://www.zacks.com/stock/chart/{ticker}/fundamental/{ratio_code}"

        if self.lang == 'vi':
            return "Xin lỗi! Không xác định được yêu cầu của bạn"
        return "Sorry! Could not figure out what the user wants"
