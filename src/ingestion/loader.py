from pathlib import Path
from pypdf import PdfReader
import docx


def load_file(path: str) -> str:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if suffix == ".docx":
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")
