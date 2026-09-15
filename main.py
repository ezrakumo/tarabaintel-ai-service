from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
from huggingface_hub import InferenceClient

app = FastAPI(title="TarabaInsight AI Microservice")

# Initialize the official HF client (handles DNS and routing automatically)
client = InferenceClient(token=os.environ.get('HUGGINGFACE_API_KEY'))

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    prompt = f"""You are an intelligence analyst for Taraba State. Analyze this report and respond in valid JSON format ONLY. Do not include any text outside the JSON.

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
        # Use the official client which handles the network routing perfectly
        response = client.chat_completion(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        
        # Clean up markdown if the LLM adds it
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
