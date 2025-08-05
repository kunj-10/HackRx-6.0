import fitz  # PyMuPDF
import docx
import email
import os
from typing import Optional
import pandas as pd


def sanitize_text(text: str) -> str:
    # Remove null characters and strip
    return text.replace("\x00", "").strip()

def extract_from_pdf(file_path: str) -> str:
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return sanitize_text(text)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

def extract_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return sanitize_text(text)
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}")

def extract_from_email(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode(errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body += payload.decode(errors="ignore")
            return sanitize_text(body)
    except Exception as e:
        raise RuntimeError(f"Email extraction failed: {e}")

def extract_from_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")
        return sanitize_text(df.to_markdown(index=False))
    except Exception as e:
        raise RuntimeError(f"CSV extraction failed: {e}")

def extract_from_xlsx(file_path: str) -> str:
    try:
        df_list = pd.read_excel(file_path, sheet_name=None) 
        text = ""
        for sheet_name, df in df_list.items():
            text += f"Sheet: {sheet_name}\n"
            text += df.to_markdown(index=False)
            text += "\n"
        return sanitize_text(text)
    except Exception as e:
        raise RuntimeError(f"XLSX extraction failed: {e}")      

def extract_fallback(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sanitize_text(f.read())
    except Exception:
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return sanitize_text(content.decode("utf-8", errors="ignore"))
        except Exception as e:
            raise RuntimeError(f"Fallback extraction failed: {e}")

def extract_text(file_path: str, mime_type: Optional[str] = None) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf" or mime_type == "application/pdf":
            return extract_from_pdf(file_path)
        elif ext == ".docx" or mime_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            return extract_from_docx(file_path)
        elif ext in [".eml", ".msg"] or mime_type == "message/rfc822":
            return extract_from_email(file_path)
        elif ext == ".csv" or mime_type == "text/csv":
            return extract_from_csv(file_path)
        elif ext == ".xlsx" or mime_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]:
            return extract_from_xlsx(file_path)
        else:
            return extract_fallback(file_path)
    except Exception as e:
        return f"Error extracting text: {str(e)}"