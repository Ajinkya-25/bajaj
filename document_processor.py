# document_processor.py
# This is the correct and final version.
# It uses the PyMuPDF (fitz) library, which is the proper tool for this task.
# The persistent "ModuleNotFoundError: No module named 'fitz'" is an
# environment setup issue that must be resolved by correctly installing
# the packages from requirements.txt.

import fitz  # PyMuPDF
import pytesseract
from docx import Document
import io
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from PIL import Image
from PIL import ImageOps
import re

class DocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, file_content: bytes) -> str:
        pass

    @abstractmethod
    def get_file_type(self) -> str:
        pass

class PDFProcessor(DocumentProcessor):
    def extract_text(self, file_content: bytes) -> str:
        """
        Extracts text from a PDF using PyMuPDF (fitz), with an OCR fallback.
        This is the most reliable method for text-based PDFs.
        """
        print("INFO: Starting PDF text extraction with PyMuPDF (fitz)...")
        text = ""
        try:
            # Use fitz to open the PDF from memory
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            print(f"INFO: Document has {pdf_document.page_count} pages.")
            
            for page_num, page in enumerate(pdf_document):
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
            
            pdf_document.close()
            print(f"INFO: PyMuPDF extracted {len(text)} characters.")

            # If the extracted text is not meaningful, it might be a scanned PDF.
            if self._is_meaningful(text):
                print("INFO: PyMuPDF extraction successful.")
                return text
            else:
                print("WARN: PyMuPDF text is not meaningful. Falling back to OCR.")
                return self._ocr_fallback(file_content)

        except Exception as e:
            print(f"ERROR: PyMuPDF failed: {e}. Falling back to OCR.")
            return self._ocr_fallback(file_content)

    def _is_meaningful(self, text: str) -> bool:
        """A check to see if the text is meaningful or just gibberish."""
        if len(text.strip()) < 1000:
            return False
        
        # Count words with 3+ English letters
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        if len(words) < 150:
            return False
            
        return True

    def _ocr_fallback(self, file_content: bytes) -> str:
        """OCR fallback using PyMuPDF to render pages and Pytesseract to read them."""
        print("INFO: Starting OCR fallback process...")
        text = ""
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num, page in enumerate(pdf_document):
                print(f"INFO: OCR processing page {page_num + 1}/{len(pdf_document)}")
                # Render page to a high-resolution image
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Pre-process image for better OCR results
                img = img.convert('L') 
                img = ImageOps.autocontrast(img)
                
                # Use Tesseract to OCR the image
                page_text = pytesseract.image_to_string(img, lang='eng')
                text += page_text + "\n"

            pdf_document.close()
            print(f"INFO: OCR fallback extracted {len(text)} characters.")
            return text
        except Exception as ocr_error:
            print(f"ERROR: OCR process failed: {ocr_error}")
            return "Failed to extract text from PDF using all available methods."

    def get_file_type(self) -> str:
        return "PDF"

class DOCXProcessor(DocumentProcessor):
    def extract_text(self, file_content: bytes) -> str:
        doc_file = io.BytesIO(file_content)
        doc = Document(doc_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def get_file_type(self) -> str:
        return "DOCX"

class PlainTextProcessor(DocumentProcessor):
    def extract_text(self, file_content: bytes) -> str:
        return file_content.decode('utf-8', errors='ignore')

    def get_file_type(self) -> str:
        return "TXT"

class DocumentProcessorFactory:
    processors = {
        '.pdf': PDFProcessor(),
        '.docx': DOCXProcessor(),
        '.txt': PlainTextProcessor(),
        '.text': PlainTextProcessor(),
    }

    @classmethod
    def get_processor(cls, filename: str) -> DocumentProcessor:
        try:
            path_part = filename.split('?')[0]
            extension = '.' + path_part.split('.')[-1].lower()
            return cls.processors.get(extension, cls.processors['.txt'])
        except:
            return cls.processors['.txt']