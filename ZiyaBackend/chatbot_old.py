import os
import json
import torch
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Gemini / GenAI configuration ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    gemini_client = genai.Client(api_key=GEMINI_KEY)
else:
    gemini_client = None
    print("Warning: GEMINI_API_KEY is not set. Gemini verification disabled.")

GEMINI_MODEL = "gemini-1.5-flash"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Normalization
def normalize_urdu(text):
    text = text.replace("ي", "ی").replace("ے", "ی").replace("ة", "ہ").replace("ه", "ہ")
    return re.sub(r'\s+', ' ', text).strip()

def get_urdu_tokens(text):
    normalized = normalize_urdu(text)
    return set(re.findall(r'[\u0600-\u06FF]+', normalized))

# 2. Load Model & Data
print("Loading Semantic Model...")
model = SentenceTransformer('intfloat/multilingual-e5-small')

DATA_PATH = "product_files/product_catalog.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

QUESTIONS = [normalize_urdu(item.get('Question', '')) for item in DATASET]
QUESTION_EMBEDDINGS = model.encode(QUESTIONS, convert_to_tensor=True)

SYSTEM_PROMPT = "Aap ek Muavin Hajj AI Assistant hain. Sirf Context se jawab dein aur Reference lazmi shamil karein."


def extract_genai_text(response):
    if not response:
        return ""
    candidates = getattr(response, "candidates", None)
    if candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts:
                first_part = parts[0]
                return getattr(first_part, "text", "") or ""
    return getattr(response, "text", "") or ""


def verify_with_gemini(prompt):
    if gemini_client is None:
        return None
    chat = gemini_client.chats.create(model=GEMINI_MODEL)
    response = chat.send_message({"text": prompt})
    return extract_genai_text(response)


# 3. Main Logic with Fallback
@app.post("/ask")
async def ask_assistant(request: Request):
    try:
        data = await request.json()
        query = normalize_urdu(data.get("query", ""))
        
        # Semantic Matching
        query_emb = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, QUESTION_EMBEDDINGS)[0]
        top_k = min(3, len(DATASET))
        top_results = torch.topk(scores, k=top_k)
        top_scores = [float(s) for s in top_results.values]
        top_idxs = [int(i) for i in top_results.indices]

        idx = top_idxs[0]
        max_score = top_scores[0]

        # Threshold Guard
        if max_score < 0.82:
            return {
                "answer": "Maazrat, iska jawab hamare scholars ke dataset mein nahi mila.",
                "score": round(max_score, 4),
                "ref": "N/A"
            }

        # Avoid a weak top match when multiple questions are very close
        if top_k > 1 and top_scores[0] - top_scores[1] < 0.04:
            query_tokens = get_urdu_tokens(query)
            matched_tokens = get_urdu_tokens(DATASET[idx].get('Question', ''))
            overlap = len(query_tokens & matched_tokens) / max(1, len(query_tokens))
            if overlap < 0.25:
                return {
                    "answer": "Maazrat, iska jawab hamare scholars ke dataset mein nahi mila.",
                    "score": round(max_score, 4),
                    "ref": "N/A"
                }

        matched_entry = DATASET[idx]
        raw_answer = matched_entry.get('Answer', '')
        reference = matched_entry.get('Reference', 'N/A')

        final_response = f"{raw_answer}\n\nReference: {reference}"
        verify_text = None
        if gemini_client is not None:
            try:
                verification_prompt = (
                    f"Does this dataset answer match the user query?"
                    f"\nUser query: \"{query}\""
                    f"\nDataset question: \"{matched_entry.get('Question', '')}\""
                    f"\nDataset answer: \"{raw_answer}\""
                    f"\nAnswer only YES or NO."
                )
                verify_text = verify_with_gemini(verification_prompt)
                verify_answer = (verify_text or "").strip().upper().splitlines()[0] if verify_text else ""

                if verify_answer.startswith("NO"):
                    return {
                        "answer": "Maazrat, iska jawab dataset mein nahi mila.",
                        "score": round(max_score, 4),
                        "ref": "N/A"
                    }
                if verify_answer and not verify_answer.startswith("YES"):
                    print(f"Gemini verification returned an ambiguous response: {verify_text}")
                else:
                    print(f"Verification result: {verify_answer}")

            except Exception as ai_error:
                print(f"Gemini verification failed: {ai_error}. Returning direct dataset answer.")

        else:
            print("Gemini verification skipped because GEMINI_API_KEY is not configured.")

        return {
            "answer": final_response,
            "score": round(max_score, 4),
            "ref": reference
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"answer": "Backend Busy hai, dobara koshish karein."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)