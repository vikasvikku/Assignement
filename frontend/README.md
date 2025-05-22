# Document Research & Theme Identifier Frontend

A Streamlit-based frontend for the Document Research & Theme Identifier application. This frontend provides an intuitive interface for uploading documents, querying the knowledge base, and visualizing identified themes.

## Features

- Document upload with support for PDF and image files
- Real-time document processing status
- Interactive query interface
- Theme visualization with treemap charts
- Detailed analysis with citations
- Responsive and modern UI

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Start the backend server first (see backend README)
2. Launch the frontend using the command above
3. Upload documents using the sidebar
4. Enter your research query in the main interface
5. View the identified themes and their visualizations

## Requirements

- Python 3.8+
- Streamlit
- Requests
- Pandas
- Plotly
- Python-dotenv

## Note

Make sure the backend server is running at `http://localhost:8000` before using the frontend application. 