import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
import time

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'themes' not in st.session_state:
        st.session_state.themes = []
    if 'document_responses' not in st.session_state:
        st.session_state.document_responses = []

def upload_documents(files: List[UploadedFile]) -> Dict:
    """Upload documents to the backend"""
    try:
        files_data = [('files', (file.name, file.getvalue())) for file in files]
        response = requests.post(f"{API_URL}/upload", files=files_data)
        response.raise_for_status()
        result = response.json()
        
        # Update session state with new documents
        if result and 'documents' in result:
            st.session_state.documents = result['documents']
            st.session_state.document_responses = []  # Clear previous responses
            st.session_state.themes = []  # Clear previous themes
            
        return result
    except Exception as e:
        st.error(f"Error uploading documents: {str(e)}")
        return None

def analyze_query(query: str) -> Dict:
    """Analyze query and get themes"""
    try:
        st.write(f"Connecting to backend at: {API_URL}")
        with st.spinner("Analyzing documents..."):
            response = requests.post(
                f"{API_URL}/analyze",
                json={"query": query},
                timeout=30  # Add timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result and 'themes' in result:
                st.session_state.themes = result['themes']
                # Update document responses based on themes
                doc_responses = []
                for theme in result['themes']:
                    for doc in theme['documents']:
                        doc_responses.append({
                            'Document ID': doc['doc_id'],
                            'Extracted Answer': doc['extracted_answer'],
                            'Citation': doc['citation'],
                            'Theme': theme['theme']
                        })
                st.session_state.document_responses = doc_responses
                st.success(f"Successfully analyzed query. Found {len(result['themes'])} themes.")
                
            return result
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not connect to backend at {API_URL}. Please ensure the backend service is running.")
        st.error(f"Error details: {str(e)}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend is taking too long to respond.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend returned an error: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response details: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error analyzing query: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Document Research & Theme Identifier",
        page_icon="📚",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Document Research")
        st.markdown("---")
        
        # Document Upload Section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    result = upload_documents(uploaded_files)
                    if result:
                        st.success(f"Successfully processed {len(uploaded_files)} documents")
        
        st.markdown("---")
        
        # Document List
        st.subheader("Processed Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.write(f"📄 {doc['filename']}")
    
    # Main Content
    st.title("Document Research & Theme Identifier")
    
    # Query Section
    st.subheader("Research Query")
    query = st.text_area("Enter your research query:", height=100)
    
    if query:
        if st.button("Analyze"):
            with st.spinner("Analyzing documents..."):
                result = analyze_query(query)
                if result:
                    st.success("Analysis complete!")
    
    # Results Section
    if st.session_state.themes:
        st.subheader("Identified Themes")
        
        # Theme Cards
        cols = st.columns(len(st.session_state.themes))
        for idx, theme in enumerate(st.session_state.themes):
            with cols[idx]:
                st.markdown(f"""
                <div style='
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                '>
                    <h3 style='
                        color: #1f1f1f;
                        font-size: 1.5em;
                        margin-bottom: 15px;
                        border-bottom: 2px solid #4a90e2;
                        padding-bottom: 8px;
                    '>{theme['theme']}</h3>
                    <p style='
                        color: #333333;
                        font-size: 1.1em;
                        line-height: 1.5;
                        margin-bottom: 15px;
                    '>{theme['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Document Responses Table
        st.subheader("Document Responses")
        doc_data = []
        for theme in st.session_state.themes:
            for doc in theme['documents']:
                doc_data.append({
                    'Document ID': doc['doc_id'],
                    'Extracted Answer': doc['extracted_answer'],
                    'Citation': doc['citation'],
                    'Theme': theme['theme']
                })
        
        if doc_data:
            df = pd.DataFrame(doc_data)
            st.dataframe(
                df,
                column_config={
                    "Document ID": st.column_config.TextColumn("Document ID", width="medium"),
                    "Extracted Answer": st.column_config.TextColumn("Extracted Answer", width="large"),
                    "Citation": st.column_config.TextColumn("Citation", width="medium"),
                    "Theme": st.column_config.TextColumn("Theme", width="medium")
                },
                hide_index=True
            )
        
        # Theme Distribution
        st.subheader("Theme Distribution")
        theme_data = []
        for theme in st.session_state.themes:
            for doc in theme['documents']:
                theme_data.append({
                    'Theme': theme['theme'],
                    'Document': doc['doc_id'],
                    'Relevance': 1.0  # Default relevance
                })
        
        if theme_data:
            df = pd.DataFrame(theme_data)
            fig = px.treemap(
                df,
                path=['Theme', 'Document'],
                values='Relevance',
                color='Theme',
                title='Theme Distribution Across Documents'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Analysis
            st.subheader("Detailed Analysis")
            for theme in st.session_state.themes:
                with st.expander(f"{theme['theme']}: {theme['description']}"):
                    st.write("**Supporting Documents:**")
                    for doc in theme['documents']:
                        st.markdown(f"""
                        <div style='
                            background-color: #f8f9fa;
                            padding: 15px;
                            border-radius: 8px;
                            margin-bottom: 10px;
                            border-left: 4px solid #4a90e2;
                        '>
                            <strong>Document {doc['doc_id']}</strong><br>
                            <em>Answer:</em> {doc['extracted_answer']}<br>
                            <em>Citation:</em> {doc['citation']}
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 