# processing/document_processor.py
import PyPDF2
from docx import Document
import io
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class DocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, file_content: bytes) -> str:
        pass

    @abstractmethod
    def get_file_type(self) -> str:
        pass


class PDFProcessor(DocumentProcessor):
    def extract_text(self, file_content: bytes) -> str:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def get_file_type(self) -> str:
        return "PDF"


class DOCXProcessor(DocumentProcessor):
    def extract_text(self, file_content: bytes) -> str:
        doc_file = io.BytesIO(file_content)
        doc = Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
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
        extension = '.' + filename.split('.')[-1].lower()
        if extension in cls.processors:
            return cls.processors[extension]
        else:
            # Default to plain text for unknown extensions
            return cls.processors['.txt']
