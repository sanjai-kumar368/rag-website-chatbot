from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

# ============================================================
# APP CONFIG
# ============================================================

app = Flask(__name__)
app.secret_key = "super-secret-key-123"

# ============================================================
# LOAD WEBSITE CONTENT
# ============================================================

with open("data/website_content.txt", "r", encoding="utf-8") as f:
    WEBSITE_CONTENT = f.read()

# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant for SkyNet Fiber.

Answer ONLY using the context below.
If the answer is not found, say:
"I do not have that information."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def generate_answer(query: str) -> str:
    response = llm.invoke(
        prompt.format(
            context=WEBSITE_CONTENT,
            question=query
        )
    )
    return response.content.strip()

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if not query:
            return redirect(url_for("chat"))

        if query.lower() in ["hi", "hello", "hey", "hii", "hai"]:
            answer = "Hello. How can I help you today?"
        else:
            answer = generate_answer(query)

        session["chat"].append(("user", query))
        session["chat"].append(("bot", answer))
        session.modified = True

    return render_template("index.html", chat=session["chat"])


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please ask a valid question."})

    answer = generate_answer(query)
    return jsonify({"answer": answer})


@app.route("/clear")
def clear_chat():
    session.pop("chat", None)
    return redirect(url_for("chat"))

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
