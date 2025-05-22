# Document Research Assistant

A web application that helps users analyze and extract themes from documents using AI.

## Features

- Document upload and processing
- Theme analysis using Groq AI
- Document preview and citation
- Interactive visualization of themes
- Vector-based document search

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   └── services/
│   │       ├── document_processor.py
│   │       └── theme_analyzer.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
└── render.yaml
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd document-research-assistant
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory with your API keys:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=your_qdrant_url
```

## Running Locally

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start the frontend:
```bash
cd frontend
streamlit run app.py
```

## Deployment

The application is configured for deployment on Render. The `render.yaml` file contains the deployment configuration for both frontend and backend services.

## Technologies Used

- Backend:
  - FastAPI
  - Groq AI
  - Langchain
  - Sentence Transformers
  - Qdrant

- Frontend:
  - Streamlit
  - Plotly
  - Pandas

## License

MIT

