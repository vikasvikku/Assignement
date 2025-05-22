import os
from typing import Dict, Any, List, Optional
import PyPDF2
from PIL import Image
import pytesseract
import io
import uuid
from datetime import datetime
import google.generativeai as genai
from ..config import settings
from ..models.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fastapi import UploadFile
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the sentence transformer model"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the sentence transformer model"""
        embedding = self.model.encode(text)
        return embedding.tolist()

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            'pdf': self._process_pdf,
            'png': self._process_image,
            'jpg': self._process_image,
            'jpeg': self._process_image,
            'tiff': self._process_image,
            'bmp': self._process_image
        }
        self.max_file_size = settings.MAX_FILE_SIZE
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        # Initialize the embedding model
        self.embedding_model = SentenceTransformerEmbeddings()

    async def process_files(self, files: List[UploadFile]) -> List[Document]:
        processed_docs = []
        for file in files:
            doc = await self._process_single_file(file)
            processed_docs.append(doc)
        return processed_docs

    async def _process_single_file(self, file: UploadFile) -> Document:
        content = ""
        if file.filename.endswith('.pdf'):
            content = self._extract_text_from_pdf(file)
        else:
            content = self._extract_text_from_image(file)

        # Generate embedding using sentence-transformers
        embedding = self.embedding_model.embed_query(content)

        doc = Document(
            doc_id=str(uuid.uuid4()),
            filename=file.filename,
            content=content,
            upload_date=datetime.now(),
            page_count=1,  # Simplified for now
            embedding=embedding
        )

        # Create vector store if not exists
        if not self.vector_store:
            self.vector_store = Chroma(
                collection_name="documents",
                embedding_function=self.embedding_model
            )
        
        # Add text to vector store
        self.vector_store.add_texts([content])

        return doc

    def _extract_text_from_pdf(self, file: UploadFile) -> str:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def _extract_text_from_image(self, file: UploadFile) -> str:
        image = Image.open(file.file)
        return pytesseract.image_to_string(image)

    async def get_all_documents(self) -> List[Document]:
        # This would typically query a database
        # For now, return empty list
        return []

    async def process_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a document and extract its content
        """
        try:
            # Validate file
            if not self.validate_file(content, filename):
                raise ValueError("Invalid file format or size")

            # Get file extension
            ext = filename.split('.')[-1].lower()
            
            # Process based on file type
            if ext in self.supported_formats:
                content = await self.supported_formats[ext](content)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Generate metadata
            metadata = {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "file_type": ext,
                "upload_date": datetime.now().isoformat(),
                "page_count": len(content),
                "paragraph_count": sum(len(page.get("paragraphs", [])) for page in content)
            }

            return {
                "content": content,
                "metadata": metadata
            }

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def validate_file(self, content: bytes, filename: str) -> bool:
        """
        Validate file format and size
        """
        # Check file size
        if len(content) > self.max_file_size:
            return False

        # Check file extension
        ext = filename.split('.')[-1].lower()
        return ext in self.supported_formats

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results using OpenCV
        """
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)

    async def _process_pdf(self, content: bytes) -> List[Dict[str, Any]]:
        """
        Process PDF file and extract text using PyPDF2
        """
        try:
            # Read PDF from bytes
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            pages = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Process text into paragraphs
                paragraphs = self._split_into_paragraphs(text)
                
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "paragraphs": paragraphs,
                    "paragraph_count": len(paragraphs)
                })
            
            return pages

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    async def _process_image(self, content: bytes) -> List[Dict[str, Any]]:
        """
        Process image file and extract text using OCR with OpenCV preprocessing
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(content))
            
            # Preprocess image using OpenCV
            processed_image = self._preprocess_image(image)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(processed_image)
            
            # Process text into paragraphs
            paragraphs = self._split_into_paragraphs(text)
            
            return [{
                "page_number": 1,
                "text": text,
                "paragraphs": paragraphs,
                "paragraph_count": len(paragraphs)
            }]

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into paragraphs and sentences
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        
        # Split into paragraphs (handle multiple newline patterns)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Process each paragraph
        processed_paragraphs = []
        for i, para in enumerate(paragraphs, 1):
            # Split into sentences (handle multiple sentence endings)
            sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
            
            # Clean sentences
            sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences]
            
            processed_paragraphs.append({
                "paragraph_number": i,
                "text": para,
                "sentences": sentences,
                "sentence_count": len(sentences)
            })
        
        return processed_paragraphs 