from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI(title="TarabaInsight AI Microservice")

# Google Gemini API Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    prompt = f"""You are an expert intelligence analyst for Taraba State, Nigeria. 
    Analyze the following citizen report and respond ONLY with a valid JSON object. Do not include markdown formatting like ```json.

    Category: {request.issue_category}
    Description: {request.description}

    Provide JSON with these exact keys:
    - "ai_suggested_category": (string) Refined category.
    - "ai_confidence_score": (float between 0.0 and 1.0) Your confidence level.
    - "sentiment": (string) "NEGATIVE", "NEUTRAL", or "POSITIVE".
    - "urgency_level": (string) "CRITICAL", "MODERATE", or "LOW".
    - "extracted_entities": (object) containing "locations" (List of strings) and "keywords" (List of strings).
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500}
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        
        result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        # Clean up if Gemini adds markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(result_text)
        
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
