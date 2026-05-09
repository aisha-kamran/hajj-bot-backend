import os
import json
import re
import shutil
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "product_files" / "product_catalog.json"
DB_PATH = str(BASE_DIR / "ziya_vector_db")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("MODEL_NAME", "gemini-1.5-flash")

QUESTION_LABEL = "سوال"
ANSWER_LABEL = "جواب"
GREETING_RESPONSE = "وعلیکم السلام! میں حج و عمرہ کے مسائل میں آپ کی رہنمائی کے لیے حاضر ہوں۔"
EMPTY_QUESTION_RESPONSE = "براہ کرم سوال لکھیں۔"
UNCLEAR_RESPONSE = "براہ کرم سوال کو مزید واضح کریں۔"
SERVER_ERROR_RESPONSE = "سسٹم میں مسئلہ آ گیا ہے، دوبارہ کوشش کریں۔"

IMPORTANT_WORDS = [
    "سعی", "مسعی", "طواف", "احرام", "عرفات", "عرفہ", "منی", "جمرہ",
    "رمی", "مزدلفہ", "حیض", "غسل", "تیمم",
]

LOCATION_WORDS = {
    "کہاں", "کدھر", "جگہ", "مقام", "location", "where", "place",
    "kahan", "kidhar", "kidr", "kaha", "أين", "اين", "مكان",
}

STOP_WORDS = {
    "کیا", "ہے", "ہیں", "میں", "سے", "کو", "کا", "کی", "کے", "اور",
    "اگر", "تو", "پر", "یہ", "وہ", "نے", "بھی", "نہیں", "کر", "کرتے",
    "کرنا", "ہو", "ہوں", "لئے", "لیے", "myn", "mein", "main", "hy",
    "hai", "kya", "kysa", "kaisa", "krna", "karna",
}

ROMAN_URDU_MAP = {
    "masjid": "مسجد",
    "nimrah": "نمرہ",
    "namirah": "نمرہ",
    "numrah": "نمرہ",
    "waqoof": "وقوف",
    "wuqoof": "وقوف",
    "wukuf": "وقوف",
    "arafat": "عرفات",
    "arfa": "عرفہ",
    "hajj": "حج",
    "umrah": "عمرہ",
    "tawaf": "طواف",
    "sai": "سعی",
    "saee": "سعی",
    "ihram": "احرام",
    "mina": "منی",
    "muzdalifa": "مزدلفہ",
    "rami": "رمی",
    "jamarat": "جمرات",
    "jamra": "جمرہ",
    "haidh": "حیض",
    "haiz": "حیض",
    "ghusal": "غسل",
    "ghusl": "غسل",
    "tayammum": "تیمم",
    "menstruating": "حیض",
    "menstruation": "حیض",
    "period": "حیض",
    "standing": "وقوف",
    "stand": "وقوف",
    "stay": "وقوف",
    "happens": "حکم",
    "ruling": "حکم",
    "outside": "خارج",
    "inside": "داخل",
    "boundary": "حدود",
    "boundaries": "حدود",
    "مسجد": "مسجد",
    "المسجد": "مسجد",
    "نمره": "نمرہ",
    "نمرة": "نمرہ",
    "نمرہ": "نمرہ",
    "وقوف": "وقوف",
    "الوقوف": "وقوف",
    "عرفات": "عرفات",
    "السعي": "سعی",
    "سعي": "سعی",
    "سعی": "سعی",
    "المسعى": "مسعی",
    "مسعى": "مسعی",
    "الصفا": "صفا",
    "الصفا": "صفا",
    "المروة": "مروہ",
    "المروه": "مروہ",
    "الحيض": "حیض",
    "حائض": "حیض",
    "الحائض": "حیض",
}

LANGUAGE_NAMES = {
    "ur": "Urdu",
    "en": "English",
    "ar": "Arabic",
}

LOCALIZED_MESSAGES = {
    "empty": {
        "ur": EMPTY_QUESTION_RESPONSE,
        "en": "Please write a question.",
        "ar": "يرجى كتابة السؤال.",
    },
    "unclear": {
        "ur": UNCLEAR_RESPONSE,
        "en": "Please make your question clearer.",
        "ar": "يرجى توضيح السؤال أكثر.",
    },
    "error": {
        "ur": SERVER_ERROR_RESPONSE,
        "en": "A system error occurred. Please try again.",
        "ar": "حدث خطأ في النظام، يرجى المحاولة مرة أخرى.",
    },
    "greeting": {
        "ur": GREETING_RESPONSE,
        "en": "Wa Alaikum Assalam! I am here to help with Hajj and Umrah questions.",
        "ar": "وعليكم السلام! أنا هنا لمساعدتك في مسائل الحج والعمرة.",
    },
}

# ---------------- EMBEDDINGS ----------------
embeddings = None
catalog_cache = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return embeddings


# ---------------- HELPERS ----------------
def load_catalog():
    global catalog_cache
    if catalog_cache is not None:
        return catalog_cache

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        catalog_cache = json.load(f)

    return catalog_cache


def clean_question(text):
    text = text.strip()
    if text.startswith(QUESTION_LABEL):
        text = text[len(QUESTION_LABEL):]
    return text.strip(":-۔ \n\t")


def normalize_query(text):
    text = clean_question(text)
    text = re.sub(r"\bsa[\u2019'`-]?i\b", "sai", text, flags=re.IGNORECASE)
    roman_words = re.findall(r"[A-Za-z]+", text.lower())
    urdu_terms = [ROMAN_URDU_MAP[word] for word in roman_words if word in ROMAN_URDU_MAP]

    for key, value in ROMAN_URDU_MAP.items():
        if re.search(r"[\u0600-\u06FF]", key) and key in text:
            urdu_terms.append(value)

    if "مسجد" in urdu_terms and "نمرہ" in urdu_terms and "وقوف" in urdu_terms:
        urdu_terms.extend(["عرفات", "عرفہ"])

    if "سعی" in text and any(word in text.lower() for word in LOCATION_WORDS):
        urdu_terms.extend(["مسعی", "صفا", "مروہ"])

    if (
        any(word in text for word in ["صفا", "الصفا"])
        and any(word in text for word in ["مروہ", "مروۃ", "المروہ", "المروة"])
        and any(word in text.lower() for word in LOCATION_WORDS)
    ):
        urdu_terms.extend(["سعی", "مسعی"])

    if urdu_terms:
        return f"{text} {' '.join(dict.fromkeys(urdu_terms))}"

    return text


@lru_cache(maxsize=4096)
def tokenize(text):
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text)
    return frozenset(
        word for word in text.split()
        if len(word) > 2 and word not in STOP_WORDS
    )


def is_greeting(text):
    greetings = ["salam", "hi", "hello", "سلام", "السلام", "assalam", "hey"]
    return any(g in text.lower() for g in greetings)


def clean_fatwa(text):
    text = re.sub(
        r"جواب\s+باسمہ\s+تعالیٰ\s+وتقدس\s+الجواب\s*:?",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_relevant(query, doc):
    query_words = tokenize(query)
    haystack = f"{doc.page_content} {doc.metadata.get('question', '')}"
    doc_words = tokenize(haystack)
    return bool(query_words and query_words & doc_words)


def get_answer_text(doc):
    return clean_fatwa(doc.metadata.get("answer") or doc.page_content)


def normalize_language(language):
    language = (language or "ur").lower()
    return language if language in LANGUAGE_NAMES else "ur"


def local_message(kind, language):
    return LOCALIZED_MESSAGES[kind][normalize_language(language)]


def normalize_request_data(data):
    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        text = data.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"query": text}

        if isinstance(parsed, dict):
            return parsed

        return {"query": text}

    return {}


def translate_answer(answer, question, language):
    language = normalize_language(language)
    if language == "ur":
        return answer

    target_language = LANGUAGE_NAMES[language]

    if not OPENAI_API_KEY:
        return translate_answer_with_gemini(answer, question, target_language)

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "You translate and lightly summarize Islamic Hajj/Umrah fatwa answers. "
                    "Use only the supplied dataset answer. Do not add new rulings, opinions, "
                    "or facts. Preserve the ruling and important conditions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Target language: {target_language}\n"
                    "Return 4-6 clear lines in the target language.\n\n"
                    f"Dataset answer:\n{answer[:4000]}"
                ),
            },
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print("OpenAI translation error:", e)
        return answer

    if data.get("output_text"):
        return data["output_text"].strip()

    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(content["text"])

    return "\n".join(texts).strip() or answer


def translate_answer_with_gemini(answer, question, target_language):
    if not GEMINI_API_KEY:
        return answer

    try:
        import google.generativeai as genai
    except ImportError as e:
        print("Gemini translation unavailable:", e)
        return answer

    prompt = (
        "Translate and lightly summarize this Hajj/Umrah fatwa answer. "
        "Use only the supplied dataset answer. Do not add any new ruling, "
        "opinion, or fact. Preserve the ruling and important conditions.\n\n"
        f"Question: {question}\n"
        f"Target language: {target_language}\n"
        "Return 4-6 clear lines in the target language.\n\n"
        f"Dataset answer:\n{answer[:4000]}"
    )

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return (getattr(response, "text", "") or "").strip() or answer
    except Exception as e:
        print("Gemini translation error:", e)
        return answer


def format_answer(answer, question, language, limit=1200):
    answer = answer[:limit]
    return translate_answer(answer, question, language)


def has_sai_location_intent(query):
    query = query.lower()
    return "سعی" in query and any(word in query for word in LOCATION_WORDS)


def has_safa_marwah_location_intent(query):
    query = query.lower()
    return (
        any(word in query for word in ["صفا", "الصفا"])
        and any(word in query for word in ["مروہ", "مروۃ", "المروہ", "المروة"])
        and any(word in query for word in LOCATION_WORDS)
    )


def has_masjid_nimrah_waqoof_intent(query):
    return all(word in query for word in ["مسجد", "نمرہ", "وقوف"])


def has_hayd_sai_intent(query):
    return "سعی" in query and any(word in query for word in ["حیض", "حائض", "حائضہ", "ماہواری"])


def has_local_concise_intent(query):
    return (
        has_sai_location_intent(query)
        or has_safa_marwah_location_intent(query)
        or has_masjid_nimrah_waqoof_intent(query)
        or has_hayd_sai_intent(query)
    )


def sentence_split(text):
    parts = re.split(r"(?<=[۔.!؟?])\s+|\n+", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def trim_to_sentence_boundary(text, max_chars):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()
    matches = list(re.finditer(r"[۔.!؟?]", clipped))
    if matches:
        return clipped[: matches[-1].end()].strip()

    return clipped.rsplit(" ", 1)[0].strip()


def extractive_summary(answer, query, max_chars=650):
    query_words = tokenize(query)
    sentences = sentence_split(answer)

    if not sentences:
        return trim_to_sentence_boundary(answer, max_chars)

    if has_sai_location_intent(query) or has_safa_marwah_location_intent(query):
        for sentence in sentences:
            if (
                ("لہٰذا معلوم ہوا" in sentence or "لہذا معلوم ہوا" in sentence)
                and any(word in sentence for word in ["مسعیٰ", "مسعٰی", "مسعی"])
            ):
                return trim_to_sentence_boundary(sentence, max_chars)

    if "زمزم" in query:
        for sentence in sentences:
            if "زمزم" in sentence:
                return trim_to_sentence_boundary(sentence, max_chars)

    scored = []
    for index, sentence in enumerate(sentences[:30]):
        sentence_words = tokenize(sentence)
        score = len(query_words & sentence_words) * 4
        if any(word in sentence for word in IMPORTANT_WORDS):
            score += 2
        if "یعنی" in sentence:
            score += 1
        if "لہٰذا معلوم ہوا" in sentence or "لہذا معلوم ہوا" in sentence:
            score += 8
        if has_sai_location_intent(query) and any(
            word in sentence for word in ["مسعیٰ", "مسعٰی", "مسعی", "مسجد سے خارج"]
        ):
            score += 5
        score -= index * 0.05
        scored.append((score, index, sentence))

    selected = [
        item for item in sorted(scored, key=lambda item: item[0], reverse=True)
        if item[0] > 0
    ][:4]

    if not selected:
        selected = scored[:3]

    selected = sorted(selected, key=lambda item: item[1])
    summary = " ".join(sentence for _, _, sentence in selected)
    return trim_to_sentence_boundary(summary, max_chars)


def concise_dataset_answer(answer, query, language="ur"):
    """
    Return a short answer that is grounded only in the matched dataset answer text.
    """
    q = query or ""
    picked = extractive_summary(answer, q, max_chars=520)

    language = normalize_language(language)
    translated = translate_answer(picked, q, language)
    return trim_to_sentence_boundary(translated or picked, 700)


def catalog_score(query, item):
    query_words = tokenize(query)
    question = clean_question(item.get("Question", ""))
    answer = item.get("Answer", "")
    text = f"{question} {answer}"

    if not query_words:
        return 0

    question_words = tokenize(question)
    answer_words = tokenize(answer)
    score = (len(query_words & question_words) * 3) + len(query_words & answer_words)

    if query and query in question:
        score += 100
    elif question and question in query:
        score += 50

    if "زمزم" in query and "زمزم" in text:
        score += 40
        if any(word in query for word in ["بئر", "معلومات", "حوالے", "بارے"]):
            score += 20 if "معلومات" in question else 5

    if "سعی" in query and "صحن" in query and "صحن" in text:
        score += 60
        if "سعی" in question and "صحن" in question:
            score += 30

    has_location_intent = (
        has_sai_location_intent(query) or has_safa_marwah_location_intent(query)
    )
    if has_location_intent:
        if "مسعی" in question:
            score += 90
        if any(word in text for word in ["مسعی", "مسعٰی", "مسعیٰ", "صفا", "مروہ"]):
            score += 40
        if any(word in question for word in ["حیض", "حائض", "افضل", "پہلے", "بعد", "تقدیم", "تاخیر", "حکم"]):
            score -= 25

    return score


def best_catalog_match(query):
    best_item = None
    best_score = 0

    for item in load_catalog():
        score = catalog_score(query, item)
        if score > best_score:
            best_item = item
            best_score = score

    query_word_count = len(tokenize(query))
    minimum_score = max(6, min(12, query_word_count * 2))
    if best_item and best_score >= minimum_score:
        return best_item

    return None


def best_catalog_match_with_terms(query, required_terms):
    """
    For certain intents we want to stay strictly grounded in the dataset.
    This helper narrows candidates to items that mention at least one required term
    (in question or answer), then falls back to the usual catalog scoring.
    """
    required_terms = [t for t in required_terms if t]
    if not required_terms:
        return best_catalog_match(query)

    best_item = None
    best_score = 0

    for item in load_catalog():
        question = clean_question(item.get("Question", ""))
        answer = item.get("Answer", "")
        haystack = f"{question} {answer}"
        if not any(term in haystack for term in required_terms):
            continue
        score = catalog_score(query, item)
        if score > best_score:
            best_item = item
            best_score = score

    return best_item


# ---------------- VECTOR STORE ----------------
def build_vectorstore():
    data = load_catalog()
    documents = []
    metadatas = []
    ids = []

    for index, item in enumerate(data):
        question = clean_question(item.get("Question", ""))
        answer = item.get("Answer", "").strip()
        reference = item.get("Reference", "").strip()

        if not question or not answer:
            continue

        item_id = str(item.get("ID", index + 1))
        documents.append(f"{QUESTION_LABEL}: {question}\n{ANSWER_LABEL}: {answer}")
        metadatas.append({
            "id": item_id,
            "question": question,
            "answer": answer,
            "reference": reference,
        })
        ids.append(item_id)

    return Chroma.from_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
        embedding=get_embeddings(),
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_vectorstore():
    if os.path.exists(DB_PATH):
        store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
        try:
            expected_count = len(load_catalog())
            actual_count = store._collection.count()
            sample_docs = store._collection.peek(1).get("documents") or []
            sample_text = sample_docs[0] if sample_docs else ""
            has_clean_text = "Ø" not in sample_text and "Ù" not in sample_text

            if actual_count == expected_count and has_clean_text:
                return store
        except Exception:
            pass

        shutil.rmtree(DB_PATH, ignore_errors=True)

    return build_vectorstore()


vectorstore = None


def get_cached_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore

# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask_bot(request: Request):
    language = "ur"
    try:
        data = normalize_request_data(await request.json())
        language = normalize_language(data.get("language", "ur"))
        original_query = clean_question(data.get("query", ""))
        user_query = normalize_query(original_query)

        if not original_query:
            return JSONResponse({"answer": local_message("empty", language)})

        if is_greeting(original_query):
            return JSONResponse({"answer": local_message("greeting", language)})

        if has_local_concise_intent(user_query):
            terms = []
            if has_sai_location_intent(user_query) or has_safa_marwah_location_intent(user_query):
                terms = ["مسعی", "مسعٰی", "مسعیٰ", "صفا", "مروہ", "سعی"]
            elif has_masjid_nimrah_waqoof_intent(user_query):
                terms = ["مسجد", "نمرہ", "عرفات", "وقوف"]
            elif has_hayd_sai_intent(user_query):
                terms = ["حیض", "حائض", "حائضہ", "ماہواری", "سعی", "مسعی"]
            catalog_match = best_catalog_match_with_terms(user_query, terms)
        else:
            catalog_match = best_catalog_match(user_query)
        if catalog_match:
            fatwa = clean_fatwa(catalog_match.get("Answer", ""))
            if "تفصیل" in user_query or "detail" in user_query.lower():
                return JSONResponse({"answer": format_answer(fatwa, original_query, language, 2000)})
            if has_local_concise_intent(user_query):
                return JSONResponse({"answer": concise_dataset_answer(fatwa, user_query, language)})
            fatwa = concise_dataset_answer(fatwa, user_query, language="ur")
            return JSONResponse({"answer": format_answer(fatwa, original_query, language)})
        else:
            fatwa = ""

        # ---------------- SEARCH ----------------
        if not fatwa:
            store = get_cached_vectorstore()
            results = store.similarity_search(user_query, k=10)
            filtered = [doc for doc in results if is_relevant(user_query, doc)]

            important_in_query = [word for word in IMPORTANT_WORDS if word in user_query]
            if important_in_query:
                filtered = [
                    doc for doc in filtered
                    if any(
                        word in f"{doc.page_content} {doc.metadata.get('question', '')}"
                        for word in important_in_query
                    )
                ]

            if not filtered:
                filtered = results[:3]

            if not filtered:
                return JSONResponse({"answer": local_message("unclear", language)})

            best_doc = filtered[0]
            fatwa = get_answer_text(best_doc)

            if "تفصیل" in user_query or "detail" in user_query.lower():
                return JSONResponse({"answer": format_answer(fatwa, original_query, language, 2000)})

            if len(fatwa) <= 900:
                return JSONResponse({"answer": format_answer(fatwa, original_query, language)})

        # ---------------- DATASET EXTRACTION ----------------
        answer = concise_dataset_answer(fatwa, user_query, language)

        if len(answer) < 20:
            return JSONResponse({"answer": local_message("unclear", language)})

        if has_local_concise_intent(user_query):
            return JSONResponse({"answer": answer})

        return JSONResponse({"answer": translate_answer(answer, original_query, language)})

    except Exception as e:
        print("Error:", e)
        return JSONResponse({"answer": local_message("error", language)}, status_code=500)


# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
