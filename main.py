from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI(title="TarabaInsight AI Microservice")

# OpenRouter Configuration (OpenAI-compatible, highly reliable)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    prompt = f"""You are an expert intelligence analyst for Taraba State, Nigeria. 
    Analyze the following citizen report and respond ONLY with a valid JSON object. Do not include markdown formatting.

    Category: {request.issue_category}
    Description: {request.description}

    Provide JSON with these exact keys:
    - "ai_suggested_category": (string) Refined category.
    - "ai_confidence_score": (float between 0.0 and 1.0)
    - "sentiment": (string) "NEGATIVE", "NEUTRAL", or "POSITIVE".
    - "urgency_level": (string) "CRITICAL", "MODERATE", or "LOW".
    - "extracted_entities": (object) containing "locations" (List of strings) and "keywords" (List of strings).
    """

    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct", # Free and highly reliable on OpenRouter
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tarabaintel-ai.onrender.com", # Required by OpenRouter
        "X-Title": "TarabaInsight AI"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        
        result_text = response.json()['choices'][0]['message']['content']
        
        # Clean up markdown if the LLM adds it
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
