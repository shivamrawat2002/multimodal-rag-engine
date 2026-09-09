import json
from openai import OpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)
client = OpenAI(api_key=settings.openai_api_key)

CLASSIFICATION_PROMPT = """
You are a query classifier for a multimodal RAG system.
Classify the user query and return JSON with exactly these keys:
- intent: one of [definition, numerical_lookup, visual_explanation, trend_analysis, cross_modal, procedural, general]
- modalities: list of modalities needed — any of [text, table, image]
- needs_exact_numbers: true/false
- complexity: one of [simple, medium, complex]

Return only valid JSON, no markdown.

Examples:
"What is photosynthesis?" → {"intent":"definition","modalities":["text"],"needs_exact_numbers":false,"complexity":"simple"}
"What was revenue in Q3 2024?" → {"intent":"numerical_lookup","modalities":["table","text"],"needs_exact_numbers":true,"complexity":"simple"}
"What does Figure 5 show?" → {"intent":"visual_explanation","modalities":["image","text"],"needs_exact_numbers":false,"complexity":"medium"}
"Compare Figure 3 with Table 2" → {"intent":"cross_modal","modalities":["image","table","text"],"needs_exact_numbers":true,"complexity":"complex"}
"""


def classify_query(query: str) -> dict:
    """
    Classify query intent and required modalities.
    Returns dict with intent, modalities, needs_exact_numbers, complexity.
    """
    try:
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        logger.info(f"Query classified: {result}")
        return result
    except Exception as e:
        logger.warning(f"Classification failed: {e} — using defaults")
        return {
            "intent": "general",
            "modalities": ["text", "table", "image"],
            "needs_exact_numbers": False,
            "complexity": "simple",
        }