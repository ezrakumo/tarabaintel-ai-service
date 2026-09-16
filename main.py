from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI(title="TarabaInsight AI Microservice")

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str
    image_base64: str | None = None

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    # 1. Build the text prompt
    text_prompt = f"""You are an expert intelligence analyst for Taraba State, Nigeria. 
    Analyze this citizen report and any attached image evidence. Respond ONLY with a valid JSON object. Do not include markdown formatting.

    Category: {request.issue_category}
    Description: {request.description}

    Provide JSON with these exact keys:
    - "ai_suggested_category": (string) Refined category.
    - "ai_confidence_score": (float between 0.0 and 1.0)
    - "sentiment": (string) "NEGATIVE", "NEUTRAL", or "POSITIVE".
    - "urgency_level": (string) "CRITICAL", "MODERATE", or "LOW".
    - "extracted_entities": (object) containing "locations" (List of strings) and "keywords" (List of strings).
    - "image_analysis": (string) Brief description of what the image shows, or "No image provided".
    """

    content = [{"type": "text", "text": text_prompt}]

    # 2. Attach image if it exists and is valid
    if request.image_base64 and len(request.image_base64) > 100:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{request.image_base64}"
            }
        })

    payload = {
        # ✅ Use the official, guaranteed-free vision model on OpenRouter
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
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
        response = requests.post(API_URL, json=payload, headers=headers, timeout=45)
        
        # ✅ If it fails, we will now see EXACTLY what OpenRouter complained about
        response.raise_for_status()
        
        result_text = response.json()['choices'][0]['message']['content']
        
        # Clean up markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(result_text)
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ OpenRouter HTTP Error: {e}")
        print(f"❌ OpenRouter Response: {response.text}") # This will tell us the exact problem!
        return {
            "error": f"OpenRouter Error: {response.text}",
            "ai_suggested_category": request.issue_category,
            "urgency_level": "MODERATE",
            "ai_confidence_score": 0.5,
            "sentiment": "NEUTRAL",
            "extracted_entities": {"locations": [], "keywords": []},
            "image_analysis": "Failed to process"
        }
    except Exception as e:
        print(f"❌ General Error: {e}")
        return {
            "error": str(e),
            "ai_suggested_category": request.issue_category,
            "urgency_level": "MODERATE",
            "ai_confidence_score": 0.5,
            "sentiment": "NEUTRAL",
            "extracted_entities": {"locations": [], "keywords": []},
            "image_analysis": "Failed to process"
        }
