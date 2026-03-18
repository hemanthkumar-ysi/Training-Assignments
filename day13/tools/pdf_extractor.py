from langchain_community.document_loaders import PyPDFLoader
import os
import zipfile
import re

def extract_text_from_docx_native(file_path):
    """
    Extracts text from a .docx file natively using zipfile + xml regex.
    Avoids external dependencies.
    """
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read('word/document.xml').decode('utf-8')
            # Match text inside <w:t> tags
            # eg <w:t>Hello World</w:t>
            texts = re.findall(r'<w:t.*?>(.*?)</w:t>', xml_content)
            
            # Combine text. Standard paragraphs usually hold their spacing well enough for extraction.
            return "\n".join(texts)
    except Exception as e:
        return f"Error extracting text from docx file {file_path}: {e}"

def extract_text_from_pdf(file_path):
    """
    Extracts text from a given file (PDF, TXT, or DOCX).
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    try:
        lower_path = file_path.lower()
        if lower_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif lower_path.endswith(".docx"):
            return extract_text_from_docx_native(file_path)
        else:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Combine all pages into a single text
            full_text = "\n\n".join([doc.page_content for doc in documents])
            return full_text
    except Exception as e:
        return f"Error extracting text from file {file_path}: {e}"


