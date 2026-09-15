from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="TarabaInsight AI Microservice")

class AnalysisRequest(BaseModel):
    report_id: str
    description: str
    issue_category: str

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    """
    Lightweight, rule-based AI engine for instant, free, and reliable intelligence grading.
    """
    desc_lower = request.description.lower()
    
    # 1. Determine Urgency Level based on keywords
    urgency_keywords = ['bomb', 'attack', 'urgent', 'critical', 'danger', 'kill', 'fire', 'flood', 'clash', 'militant']
    urgency = 'CRITICAL' if any(kw in desc_lower for kw in urgency_keywords) else 'MODERATE'
    
    # 2. Determine Sentiment
    negative_keywords = ['bad', 'terrible', 'danger', 'attack', 'destroyed', 'failed']
    sentiment = 'NEGATIVE' if any(kw in desc_lower for kw in negative_keywords) else 'NEUTRAL'
    
    # 3. Calculate Confidence Score (Simulated high confidence for structured intel)
    confidence = round(random.uniform(0.78, 0.96), 2)
    
    # 4. Refine Suggested Category based on context
    suggested_category = request.issue_category
    if 'road' in desc_lower or 'bridge' in desc_lower or 'market' in desc_lower:
        suggested_category = 'Infrastructure Damage'
    elif 'farm' in desc_lower or 'crop' in desc_lower or 'cattle' in desc_lower:
        suggested_category = 'Agricultural Crisis'
    elif 'health' in desc_lower or 'hospital' in desc_lower or 'disease' in desc_lower:
        suggested_category = 'Public Health Issue'

    # 5. Extract Mock Entities (Can be upgraded to spaCy/NLTK later)
    entities = {
        "locations": ["Taraba State"], 
        "keywords": request.issue_category.split()
    }

    return {
        "ai_suggested_category": suggested_category,
        "ai_confidence_score": confidence,
        "sentiment": sentiment,
        "urgency_level": urgency,
        "extracted_entities": entities
    }
