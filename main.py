from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI(title="TarabaInsight AI Microservice")

# Hugging Face API endpoint (using Phi-3-mini, a powerful free model)
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    prompt = f"""You are an intelligence analyst for Taraba State. Analyze this report and respond in valid JSON format ONLY:

Category: {request.issue_category}
Description: {request.description}

Provide JSON with these exact keys:
- "ai_suggested_category": (string) refined category
- "ai_confidence_score": (float between 0.0 and 1.0)
- "sentiment": (string) "NEGATIVE", "NEUTRAL", or "POSITIVE"
- "urgency_level": (string) "CRITICAL", "MODERATE", or "LOW"
- "extracted_entities": (object) with "locations" (array) and "keywords" (array)

Example output:
{{
  "ai_suggested_category": "Security Threat",
  "ai_confidence_score": 0.92,
  "sentiment": "NEGATIVE",
  "urgency_level": "CRITICAL",
  "extracted_entities": {{
    "locations": ["Jalingo", "market"],
    "keywords": ["militants", "attack"]
  }}
}}"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.3, "return_full_text": False}},
            timeout=15
        )
        
        result = response.json()[0]['generated_text']
        
        # Clean up the response to extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        return json.loads(result)
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            "error": str(e),
            "ai_suggested_category": request.issue_category,
            "urgency_level": "MODERATE",
            "ai_confidence_score": 0.5,
            "sentiment": "NEUTRAL",
            "extracted_entities": {"locations": [], "keywords": []}
        }
