from typing import List, Dict, Any
import google.generativeai as genai
from groq import Groq
from ..config import get_settings
import json

settings = get_settings()

class ThemeIdentifier:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Initialize Groq
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

    async def identify_themes(self, text: str, document_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify themes in the text using both Gemini and Groq models
        """
        try:
            # Create document reference map
            doc_map = {doc["doc_id"]: doc["filename"] for doc in document_responses}
            
            # Process with Gemini
            gemini_prompt = f"""
            Analyze the following text and identify the main themes. For each theme:
            1. Provide a clear, concise name
            2. Write a detailed description explaining the theme
            3. List the specific document IDs that support this theme
            4. Explain how each document supports the theme

            Return the response in this exact JSON format:
            {{
                "themes": [
                    {{
                        "name": "theme name",
                        "description": "detailed theme description",
                        "supporting_docs": [
                            {{
                                "doc_id": "DOC001",
                                "explanation": "how this document supports the theme"
                            }}
                        ]
                    }}
                ],
                "confidence": 0.85
            }}

            Text to analyze:
            {text}

            Available documents:
            {json.dumps(doc_map, indent=2)}
            """
            
            gemini_response = self.gemini_model.generate_content(gemini_prompt)
            gemini_themes = self._parse_gemini_response(gemini_response.text)

            # Process with Groq
            groq_prompt = f"""
            Analyze the following text and identify the main themes. For each theme:
            1. Provide a clear, concise name
            2. Write a detailed description explaining the theme
            3. List the specific document IDs that support this theme
            4. Explain how each document supports the theme

            Return the response in this exact JSON format:
            {{
                "themes": [
                    {{
                        "name": "theme name",
                        "description": "detailed theme description",
                        "supporting_docs": [
                            {{
                                "doc_id": "DOC001",
                                "explanation": "how this document supports the theme"
                            }}
                        ]
                    }}
                ],
                "confidence": 0.85
            }}

            Text to analyze:
            {text}

            Available documents:
            {json.dumps(doc_map, indent=2)}
            """
            
            groq_response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": groq_prompt}]
            )
            groq_themes = self._parse_groq_response(groq_response.choices[0].message.content)

            # Calculate model agreement
            agreement = self._calculate_agreement(
                [t["name"] for t in gemini_themes["themes"]],
                [t["name"] for t in groq_themes["themes"]]
            )
            
            # Combine results
            combined_themes = []
            seen_themes = set()
            
            for theme in gemini_themes["themes"] + groq_themes["themes"]:
                if theme["name"].lower() not in seen_themes:
                    seen_themes.add(theme["name"].lower())
                    # Ensure supporting_docs is a list of objects
                    if isinstance(theme.get("supporting_docs"), list):
                        theme["supporting_docs"] = [
                            doc if isinstance(doc, dict) else {"doc_id": doc, "explanation": "Supports theme"}
                            for doc in theme["supporting_docs"]
                        ]
                    else:
                        theme["supporting_docs"] = []
                    combined_themes.append(theme)
            
            return {
                "themes": combined_themes,
                "model_agreement": agreement,
                "confidence": (gemini_themes["confidence"] + groq_themes["confidence"]) / 2
            }

        except Exception as e:
            # Return a default response in case of error
            return {
                "themes": [
                    {
                        "name": "General",
                        "description": "General discussion",
                        "supporting_docs": []
                    }
                ],
                "model_agreement": 0.5,
                "confidence": 0.5
            }

    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini model response
        """
        try:
            # Extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "themes": data.get("themes", []),
                    "confidence": float(data.get("confidence", 0.5))
                }
            return {"themes": [], "confidence": 0.5}
        except Exception:
            return {"themes": [], "confidence": 0.5}

    def _parse_groq_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Groq model response
        """
        try:
            # Extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "themes": data.get("themes", []),
                    "confidence": float(data.get("confidence", 0.5))
                }
            return {"themes": [], "confidence": 0.5}
        except Exception:
            return {"themes": [], "confidence": 0.5}

    def _calculate_agreement(self, themes1: List[str], themes2: List[str]) -> float:
        """
        Calculate agreement between two sets of themes
        """
        if not themes1 or not themes2:
            return 0.0
        
        # Convert to sets for comparison
        set1 = set(t.lower() for t in themes1)
        set2 = set(t.lower() for t in themes2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0 