from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI(title="TarabaInsight AI Microservice")

# OpenRouter Configuration (Using Gemini 1.5 Flash for reliable Text + Vision)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str
    image_base64: str | None = None # ✅ Now accepts images!

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    # Build the message content (Text + Optional Image)
    content = [
        {
            "type": "text",
            "text": f"""You are an expert intelligence analyst for Taraba State, Nigeria. 
            Analyze this citizen report and any attached image evidence. Respond ONLY with a valid JSON object.

            Category: {request.issue_category}
            Description: {request.description}

            Provide JSON with these exact keys:
            - "ai_suggested_category": (string) Refined category based on text AND image.
            - "ai_confidence_score": (float between 0.0 and 1.0)
            - "sentiment": (string) "NEGATIVE", "NEUTRAL", or "POSITIVE".
            - "urgency_level": (string) "CRITICAL", "MODERATE", or "LOW".
            - "extracted_entities": (object) containing "locations" (List of strings) and "keywords" (List of strings).
            - "image_analysis": (string) Brief description of what the image shows, or "No image provided".
            """
        }
    ]

    # ✅ Attach image to the prompt if it exists
    if request.image_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{request.image_base64}"
            }
        })

    payload = {
        "model": "google/gemini-1.5-flash", # Excellent free-tier vision model on OpenRouter
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tarabaintel-ai.onrender.com",
        "X-Title": "TarabaInsight AI"
    }

    try:
        # Increased timeout to 45s to allow time for image processing
        response = requests.post(API_URL, json=payload, headers=headers, timeout=45)
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
            "extracted_entities": {"locations": [], "keywords": []},
            "image_analysis": "Failed to process image"
        }
