import sys

from fastapi.testclient import TestClient
import chatbot

sys.stdout.reconfigure(encoding="utf-8")

client = TestClient(chatbot.app)


def fake_translate(answer, question, language):
    if language == "ur":
        return answer
    return f"[{language}] {answer}"


if __name__ == "__main__":
    chatbot.translate_answer = fake_translate

    checks = [
        {
            "language": "ur",
            "query": "سعی کہاں ہوتی ہے؟",
            "expected": ["مسع", "مسجد سے خارج"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "صفا اور مروہ کہاں واقع ہیں؟",
            "expected": ["مسع", "مسجد سے خارج"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "سعی میں ایک چکر سے کیا مراد ہے؟",
            "expected": ["صفا", "مروہ"],
            "forbidden": ["سیل کبیر", "وادی محرم"],
        },
        {
            "language": "ur",
            "query": "حالت حیض میں سعی کا کیا حکم ہے؟",
            "expected": ["سعی", "طواف"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "متمتع قربانی سے پہلے حلق کروا لے تو کیا حکم ہے؟",
            "expected": ["دم", "حلق"],
            "forbidden": ["مسعی", "صفا"],
        },
        {
            "language": "ur",
            "query": "رمی قربانی حلق اور طواف زیارت میں ترتیب کا حکم کیا ہے؟",
            "expected": ["طواف", "حلق"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "کیا حدیبیہ تنعیم اور جعرانہ حدود حرم میں ہیں؟",
            "expected": ["جعرانہ", "حرم"],
            "forbidden": ["صفا", "مروہ"],
        },
        {
            "language": "ur",
            "query": "حجر اسود کے مقابل تکبیر کے ساتھ ہاتھ اٹھانا کیسا ہے؟",
            "expected": ["ہاتھ", "سنّت"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "صفا ومروہ پر ہاتھ اٹھا کر دعا مانگنا کیسا ہے؟",
            "expected": ["صفا", "مروہ", "دعا"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "ur",
            "query": "بئر زمزم کے بارے میں معلومات بتائیں",
            "expected": ["زمزم"],
            "forbidden": ["صفا", "مروہ"],
        },
        {
            "language": "ur",
            "query": "سعی کے ساتھ موجود صحن سے سعی کرنے کا حکم؟",
            "expected": ["سعی", "معتبرنہ ہوگی"],
            "forbidden": ["سیل کبیر", "میقات"],
        },
        {
            "language": "en",
            "query": "Where is Sai performed?",
            "expected": ["[en]", "مسع"],
            "forbidden": ["According to the dataset", "سیل کبیر"],
        },
        {
            "language": "ar",
            "query": "أين يكون السعي؟",
            "expected": ["[ar]", "مسع"],
            "forbidden": ["سیل کبیر", "ميقات"],
        },
    ]

    for check in checks:
        language = check["language"]
        query = check["query"]
        response = client.post("/ask", json={"query": query, "language": language})
        data = response.json()
        answer = data.get("answer", "")

        print("Language:", language, flush=True)
        print("Query:", query, flush=True)
        print("Status Code:", response.status_code, flush=True)
        print("Answer:", answer, flush=True)
        print(flush=True)

        assert response.status_code == 200
        for term in check["expected"]:
            assert term in answer
        for term in check["forbidden"]:
            assert term not in answer

    raw_string_response = client.post("/ask", json="Where is Sai performed?")
    assert raw_string_response.status_code == 200
    assert "مسع" in raw_string_response.json().get("answer", "")
