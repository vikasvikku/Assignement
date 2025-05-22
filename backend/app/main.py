from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import hashlib
import google.generativeai as genai
import groq
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pytesseract
from PIL import Image
import PyPDF2
import io
import tempfile
from app.config import Settings
from datetime import datetime
import numpy as np
import logging
import time
import threading
from .services.document_processor import DocumentProcessor, SentenceTransformerEmbeddings
from .services.theme_analyzer import ThemeAnalyzer
from .models.document import Document
import uvicorn
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Initialize Gemini
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise

# Initialize Groq
try:
    groq_client = groq.Client(api_key=settings.GROQ_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Groq: {str(e)}")
    raise

# Initialize embedding model
embedding_model = SentenceTransformerEmbeddings()

def init_qdrant():
    """Initialize Qdrant client and create collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if "documents" not in collection_names:
            logger.info("Creating collection documents")
            # Create collection with proper configuration
            qdrant_client.create_collection(
                collection_name="documents",
                vectors_config={
                    "size": 1536,  # Dimension for text-embedding-ada-002
                    "distance": "Cosine"
                },
                optimizers_config={
                    "default_segment_number": 2,
                    "max_optimization_threads": 4
                }
            )
            
            # Create payload indexes
            try:
                qdrant_client.create_payload_index(
                    collection_name="documents",
                    field_name="metadata",
                    field_schema="keyword"
                )
                qdrant_client.create_payload_index(
                    collection_name="documents",
                    field_name="text",
                    field_schema="text"
                )
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Failed to create payload indexes: {str(e)}")
                    
        logger.info("Qdrant initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")
        raise

# Initialize Qdrant client
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Document Research & Theme Identifier")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()
theme_analyzer = ThemeAnalyzer()

@app.get("/health")
def health_check():
    try:
        # Check Qdrant connection
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        # Check if documents collection exists
        if "documents" not in collection_names:
            logger.warning("Documents collection not found")
            return {"status": "warning", "message": "Documents collection not found"}
            
        # Get collection info without strict validation
        collection_info = qdrant_client.get_collection("documents")
        
        return {
            "status": "healthy",
            "qdrant": {
                "status": "connected",
                "collections": collection_names
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/test")
def test_endpoint():
    return {"status": "success", "message": "Test endpoint is working"}

# Define request models
class QueryRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = []

class AnalyzeRequest(BaseModel):
    query: str

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings for text using sentence-transformers."""
    try:
        # Generate embedding using the sentence transformer model
        return embedding_model.embed_query(text)
    except Exception as e:
        # Fallback to random embeddings if API fails
        logger.error(f"Error generating embeddings: {str(e)}")
        return np.random.rand(384).tolist()  # all-MiniLM-L6-v2 has 384 dimensions

def generate_point_id(doc_id: str, page_number: int) -> int:
    """
    Generate a unique integer ID for a point based on document ID and page number
    """
    # Create a unique string by combining doc_id and page_number
    unique_string = f"{doc_id}_{page_number}"
    # Generate a hash and convert to integer
    hash_object = hashlib.md5(unique_string.encode())
    # Take first 8 bytes of the hash and convert to integer
    return int.from_bytes(hash_object.digest()[:8], byteorder='big')

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    processed_docs = await document_processor.process_files(files)
    return {"message": f"Successfully processed {len(processed_docs)} documents", "documents": processed_docs}

@app.post("/analyze")
async def analyze_documents(request: AnalyzeRequest):
    """Analyze documents and identify themes"""
    themes = await theme_analyzer.analyze(request.query)
    return {"themes": themes}

@app.get("/documents")
async def get_documents():
    """Get list of processed documents"""
    return await document_processor.get_all_documents()

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        query = request.query
        doc_ids = request.doc_ids
        
        # If no specific documents selected, use all documents
        if not doc_ids or len(doc_ids) == 0:
            # Get all document IDs from the collection
            response = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            all_docs = {}
            for point in response[0]:
                doc_id = point.payload.get("doc_id")
                if doc_id and doc_id not in all_docs:
                    all_docs[doc_id] = {
                        "filename": point.payload.get("filename", "Unknown"),
                        "pages": []
                    }
            doc_ids = list(all_docs.keys())
        
        # Process query against each document
        doc_results = {}
        for doc_id in doc_ids:
            # Get all pages for this document
            response = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=1000,
                scroll_filter=models.Filter( 
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False
            )
            
            pages = []
            for point in response[0]:
                pages.append({
                    "page_number": point.payload.get("page_number", 0),
                    "text": point.payload.get("text", ""),
                    "paragraphs": point.payload.get("paragraphs", [])
                })
            
            # Sort pages by page number
            pages.sort(key=lambda x: x["page_number"])
            
            doc_results[doc_id] = {
                "filename": response[0][0].payload.get("filename", "Unknown") if response[0] else "Unknown",
                "pages": pages
            }
        
        # Process each document
        document_responses = []
        for doc_id, doc_data in doc_results.items():
            # Combine all pages for this document
            combined_text = "\n".join([page["text"] for page in doc_data["pages"]])
            
            # Process with Gemini for individual document response
            prompt = f"""
            Based on the following document excerpt, answer the query: {query}
            
            Document excerpt:
            {combined_text}
            
            Please provide:
            1. A direct answer that is relevant to the query
            2. The specific locations (page and paragraph numbers) where this information was found
            
            Format your response as:
            ANSWER: [your direct answer here]
            CITATION: Page [X], Paragraph [Y]
            """
            
            response = gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Extract answer and citation
            answer = ""
            citation = ""
            
            if "ANSWER:" in response_text and "CITATION:" in response_text:
                answer_part = response_text.split("CITATION:")[0].strip()
                answer = answer_part.replace("ANSWER:", "").strip()
                
                citation_part = response_text.split("CITATION:")[1].strip()
                citation = citation_part
            else:
                # Fallback if the model didn't follow the format
                answer = response_text
                citation = "Not specified"
            
            # Calculate relevance score (simple implementation)
            relevance_score = 0.5  # Default medium relevance
            if len(answer) > 20:  # If answer is substantial
                relevance_score = 0.8
            
            document_responses.append({
                "doc_id": doc_id,
                "filename": doc_data["filename"],
                "extracted_answer": answer,
                "citation": citation,
                "relevance_score": relevance_score
            })
        
        # Sort responses by relevance
        document_responses.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Combine all document responses for theme analysis
        combined_text = "\n".join([f"Document {doc['doc_id']}: {doc['extracted_answer']}" for doc in document_responses])
        
        # Identify themes using Gemini
        theme_prompt = f"""
        Analyze the following document responses to a query: "{query}"
        
        Document responses:
        {combined_text}
        
        Identify 2-4 common themes across these documents. For each theme:
        1. Provide a clear theme name
        2. Give a brief description of the theme
        3. List which documents (by doc_id) support this theme
        
        Format your response as:
        Theme 1 – [Theme Name]:
        [Theme description]
        Documents ([list of doc_ids]) [brief explanation]
        
        Theme 2 – [Theme Name]:
        [Theme description]
        Documents ([list of doc_ids]) [brief explanation]
        
        And so on for each theme.
        """
        
        theme_response = gemini_model.generate_content(theme_prompt)
        theme_text = theme_response.text
        
        # Parse themes
        themes = []
        theme_sections = theme_text.split("Theme ")
        
        for section in theme_sections:
            if not section.strip():
                continue
                
            # Extract theme number, name, and content
            parts = section.split(":", 1)
            if len(parts) < 2:
                continue
                
            theme_header = parts[0].strip()
            theme_content = parts[1].strip()
            
            # Extract theme name
            theme_name = ""
            if "–" in theme_header:
                theme_name = theme_header.split("–", 1)[1].strip()
            else:
                theme_name = theme_header
            
            # Extract supporting documents
            supporting_docs = []
            if "Documents (" in theme_content:
                docs_part = theme_content.split("Documents (")[1].split(")")[0]
                supporting_docs = [doc.strip() for doc in docs_part.split(",")]
            
            # Extract description
            description = theme_content
            if "Documents (" in description:
                description = description.split("Documents (")[0].strip()
            
            themes.append({
                "name": theme_name,
                "description": description,
                "supporting_docs": supporting_docs
            })
        
        return {
            "query": query,
            "document_responses": document_responses,
            "themes": themes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/document-content")
async def get_document_content(doc_id: str, page: int = 1):
    """Get content of a specific document page."""
    try:
        # Query the vector database for the document content
        results = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(  # Changed from 'filter' to 'scroll_filter'
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="page_number", match=models.MatchValue(value=page))
                ]
            ),
            limit=1
        )
        
        if not results[0]:
            return {"error": "Document page not found"}
        
        point = results[0][0]
        
        return {
            "doc_id": doc_id,
            "page": page,
            "content": {
                "text": point.payload.get("text", ""),
                "paragraphs": point.payload.get("paragraphs", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document content: {str(e)}")

@app.on_event("startup")
def startup_event():
    """Initialize services on startup."""
    try:
        init_qdrant()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
