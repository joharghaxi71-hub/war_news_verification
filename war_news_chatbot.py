# war_news_chatbot.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- CONFIG ----------------
GROQ_API_KEY = "enter your api key"
client = Groq(api_key=GROQ_API_KEY)

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

memory = []
timeline = []

# ---------------- MEMORY ----------------
def add_memory(role, text):
    memory.append({"role": role, "content": text})
    if len(memory) > 10:
        memory.pop(0)

# ---------------- PROMPT INJECTION ----------------
danger_patterns = [
    "ignore previous instructions",
    "system prompt",
    "override rules",
    "act as system"
]

def detect_injection(text):
    for p in danger_patterns:
        if p in text.lower():
            return True
    return False

# ---------------- NEWS SCRAPER ----------------
def scrape_news():
    urls = [
        "https://www.reuters.com/world/",
        "https://www.bbc.com/news/world"
    ]
    articles = []
    for url in urls:
        try:
            page = requests.get(url, timeout=5)
            soup = BeautifulSoup(page.text, "html.parser")
            for p in soup.find_all("p"):
                text = p.text.strip()
                if len(text) > 60:
                    articles.append(text)
        except:
            pass
    return articles[:50]

# ---------------- LOAD NEWS ----------------
documents = scrape_news()
if len(documents) == 0:
    documents = [
        "Reuters reports clashes near the border with no confirmed casualty numbers.",
        "BBC reports satellite images show troop movements but not confirmed attacks.",
        "Military analysts say battlefield claims on social media are often exaggerated."
    ]

# ---------------- EMBEDDINGS ----------------
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

# ---------------- RETRIEVAL AGENT ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def evidence_agent(claim, k=5):
    q_emb = embed_model.encode([claim], convert_to_numpy=True)[0]
    sims = [cosine_similarity(q_emb, doc_emb) for doc_emb in doc_embeddings]
    top_k_idx = np.argsort(sims)[-k:][::-1]
    results = [documents[i] for i in top_k_idx]
    scores = [sims[i] for i in top_k_idx]
    return results, scores

# ---------------- PROPAGANDA AGENT ----------------
def propaganda_agent(text):
    propaganda_words = [
        "shocking", "massive", "unbelievable",
        "secret attack", "hidden truth",
        "government lies"
    ]
    score = sum(1 for w in propaganda_words if w in text.lower())
    if score >= 2:
        return "Possible propaganda language detected"
    else:
        return "No strong propaganda patterns"

# ---------------- TIMELINE AGENT ----------------
def update_timeline(claim):
    timeline.append(claim)
    if len(timeline) > 20:
        timeline.pop(0)

# ---------------- CONFIDENCE ----------------
def confidence_score(scores):
    if len(scores) == 0:
        return 0
    avg = sum(scores) / len(scores)
    conf = max(0, min(100, int(avg * 100)))
    return conf

# ---------------- FACT CHECK ----------------
def fact_checker(claim, evidence):
    context = "\n".join(evidence)
    prompt = f"""
You are an AI war news verification system.

Claim:
{claim}

Evidence:
{context}

Task:
Determine whether the claim is TRUE, FALSE, MISLEADING, or UNVERIFIED.

Rules:
Only rely on evidence.
If evidence is weak say UNVERIFIED.
Explain briefly.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # use a model your Groq API key can access
        messages=[
            {"role":"system","content":"You are a war news fact checker."},
            {"role":"user","content":prompt}
        ]
    )
    return response.choices[0].message.content

# ---------------- VERIFY CLAIM ----------------
def verify_claim(claim):
    evidence, scores = evidence_agent(claim)
    propaganda = propaganda_agent(claim)
    verdict = fact_checker(claim, evidence)
    conf = confidence_score(scores)
    update_timeline(claim)
    return verdict, evidence, propaganda, conf

# ---------------- FASTAPI SETUP ----------------
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# POST request for chat
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.message
    if detect_injection(user_input):
        return {"verdict": "Prompt injection detected", "evidence": [], "propaganda": "Blocked", "confidence": 0}
    add_memory("user", user_input)
    verdict, evidence, propaganda, conf = verify_claim(user_input)
    add_memory("assistant", verdict)
    return {
        "verdict": verdict,
        "evidence": evidence,
        "propaganda": propaganda,
        "confidence": conf
    }

# Serve frontend HTML directly
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>War News Verification Chatbot</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
.chat-container { width: 400px; background: #fff; box-shadow: 0 8px 16px rgba(0,0,0,0.2); border-radius: 12px; display: flex; flex-direction: column; }
.messages { flex: 1; padding: 16px; overflow-y: auto; }
.message { margin-bottom: 12px; padding: 10px 14px; border-radius: 10px; max-width: 80%; }
.user { background: #e1ffc7; align-self: flex-end; }
.bot { background: #f0f0f0; align-self: flex-start; }
.input-area { display: flex; border-top: 1px solid #ddd; }
input { flex: 1; border: none; padding: 12px; font-size: 16px; border-radius: 0 0 0 12px; outline: none; }
button { border: none; background: #4a90e2; color: #fff; padding: 12px 16px; font-size: 16px; cursor: pointer; border-radius: 0 0 12px 0; }
button:hover { background: #357ab7; }
</style>
</head>
<body>
<div class="chat-container">
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <input type="text" id="userInput" placeholder="Type your message here..." />
    <button onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value.trim();
  if (!msg) return;
  appendMessage(msg, "user");
  input.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    let botMsg = data.verdict + "\\nPropaganda: " + data.propaganda + "\\nConfidence: " + data.confidence + "%\\nEvidence:\\n";
    data.evidence.forEach(e => { botMsg += "- " + e + "\\n"; });
    appendMessage(botMsg, "bot");
  } catch (err) {
    appendMessage("Error: " + err.message, "bot");
  }
}

function appendMessage(text, type) {
  const msgDiv = document.createElement("div");
  msgDiv.className = "message " + type;
  msgDiv.textContent = text;
  document.getElementById("messages").appendChild(msgDiv);
  document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
}
</script>
</body>
</html>
"""
