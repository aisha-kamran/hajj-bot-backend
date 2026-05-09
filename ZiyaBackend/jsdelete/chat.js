// js/chat.js
const chatBox = document.getElementById("chatBox");
const queryInput = document.getElementById("query");

// 🌍 Default language
let selectedLang = "ur"; // change dynamically if needed

// 👉 OPTIONAL: language buttons use karna ho to
function setLanguage(lang) {
    selectedLang = lang;
}


function addMessage(text, sender) {
    const msg = document.createElement("div");
    msg.className = `msg ${sender}`;
    msg.innerHTML = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addLoader() {
    const loader = document.createElement("div");
    loader.className = "msg bot loader";
    loader.innerHTML = "<span></span><span></span><span></span>";
    chatBox.appendChild(loader);
    chatBox.scrollTop = chatBox.scrollHeight;
    return loader;
}

function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    addMessage(query, "user");
    queryInput.value = "";

    const loader = addLoader();
    const API_URL = "https://jaylah-toothy-vernon.ngrok-free.dev";
    fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {"Content-Type": "application/json","ngrok-skip-browser-warning": "true"},
        body: JSON.stringify({ 
            query,
            language: selectedLang
            })
    })
    .then(res => res.json())
    .then(data => {
        loader.remove();
        addMessage(data.answer, "bot");
        speak(data.answer); // call TTS
    })
    .catch(err => {
        loader.remove();
        addMessage("Oops! Something went wrong. Please try again.", "bot");
        speak("Oops! Something went wrong. Please try again.");
    });
}

queryInput.addEventListener("keypress", e => {
    if (e.key === "Enter") sendQuery();
});
