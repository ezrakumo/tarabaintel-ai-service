from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
import json

app = FastAPI(title="TarabaInsight AI Microservice")

# Initialize Groq client using the API key from Render Environment Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    """
    Advanced AI Engine using Groq (Llama 3) for deep intelligence analysis.
    """
    prompt = f"""
    You are an expert intelligence analyst for Taraba State, Nigeria. 
    Analyze the following citizen report:
    
    Category: {request.issue_category}
    Description: {request.description}

    Provide your analysis in STRICT JSON format with these exact keys:
    - "ai_suggested_category": (String) Refined category (e.g., Security Threat, Agricultural Crisis).
    - "ai_confidence_score": (Float between 0.0 and 1.0) Your confidence level.
    - "sentiment": (String) "NEGATIVE", "NEUTRAL", or "POSITIVE".
    - "urgency_level": (String) "CRITICAL", "MODERATE", or "LOW".
    - "extracted_entities": (Object) containing "locations" (List of strings) and "keywords" (List of strings).

    Do not include markdown formatting like ```json, just output the raw JSON object.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile", # Groq's most stable, long-term supported model
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_content = chat_completion.choices[0].message.content
        
        # Clean up any accidental markdown formatting from the LLM
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()
            
        return json.loads(response_content)

    except Exception as e:
        return {"error": str(e), "ai_suggested_category": request.issue_category, "urgency_level": "MODERATE"}
