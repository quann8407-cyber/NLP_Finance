import re
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert_pdf_to_txt(path: str) -> str:
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    with open(path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp, set(), check_extractable=True):
            interpreter.process_page(page)

    text = retstr.getvalue()
    device.close()
    retstr.close()
    return text


def clean_pdf_text(raw_text: str) -> list:
    lines = raw_text.splitlines()
    lines = [l for l in lines if l.strip() not in ('', ' ')]
    cleaned = [re.sub(r"[^a-zA-Z0-9ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềểếễệỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]+", " ", l)
               for l in lines]
    return cleaned


def pdf_stats(docs: list) -> dict:
    """Thống kê nhanh số trang/dòng/từ - dùng để in log gọn gàng thay vì
    in cả trăm dòng text ra console."""
    total_words = sum(len(d.split()) for d in docs)
    return {
        "Số dòng văn bản": len(docs),
        "Tổng số từ": total_words,
        "Trung bình từ/dòng": round(total_words / len(docs), 1) if docs else 0,
    }
