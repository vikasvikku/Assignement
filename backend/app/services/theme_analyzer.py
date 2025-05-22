from typing import List, Dict
import groq
import json
from ..config import settings

class ThemeAnalyzer:
    def __init__(self):
        self.client = groq.Client(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    async def analyze(self, query: str) -> List[Dict]:
        try:
            # Create the prompt for theme analysis
            prompt = f"""
            Analyze the following query and identify common themes.
            
            Query: {query}
            
            Please identify 2-4 main themes and provide:
            1. Theme name
            2. Brief description
            3. Supporting citations
            
            Format your response as a JSON array of objects with the following structure:
            [
                {{
                    "theme": "Theme name",
                    "description": "Theme description",
                    "documents": [
                        {{
                            "doc_id": "DOC1",
                            "extracted_answer": "The specific answer found in this document",
                            "citation": "Page 1, Paragraph 2"
                        }}
                    ]
                }}
            ]
            
            Return ONLY the JSON array, nothing else.
            """
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing documents and identifying themes. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Parse the response as JSON
            try:
                themes = json.loads(response.choices[0].message.content)
                return themes
            except json.JSONDecodeError:
                print("Error parsing JSON response, returning mock themes")
                return self._get_mock_themes()
            
        except Exception as e:
            print(f"Error in theme analysis: {str(e)}")
            return self._get_mock_themes()
    
    def _get_mock_themes(self) -> List[Dict]:
        """Return mock themes in case of error"""
        return [
            {
                "theme": "Sample Theme 1",
                "description": "Description of theme 1",
                "documents": [
                    {
                        "doc_id": "DOC1",
                        "extracted_answer": "Sample answer from document 1",
                        "citation": "Page 1, Paragraph 2"
                    }
                ]
            },
            {
                "theme": "Sample Theme 2",
                "description": "Description of theme 2",
                "documents": [
                    {
                        "doc_id": "DOC2",
                        "extracted_answer": "Sample answer from document 2",
                        "citation": "Page 1, Paragraph 1"
                    }
                ]
            }
        ] 