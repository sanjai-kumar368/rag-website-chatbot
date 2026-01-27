from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
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
    text = f.read()

# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# GROQ LLM (FIXED MODEL)
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant for SkyNet Fiber.

Answer ONLY using the context below.
If the answer is not present, say clearly that you do not have that information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ============================================================
# VECTOR STORE (FAISS)
# ============================================================

FAISS_PATH = "faiss_index"
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

if os.path.exists(FAISS_PATH):
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================================
# HELPER FUNCTION
# ============================================================

def generate_answer(query: str) -> str:
    docs = retriever.invoke(query)

    if not docs:
        return "I can answer only SkyNet Fiber related questions."

    context = "\n\n".join(d.page_content for d in docs)

    response = llm.invoke(
        prompt.format(
            context=context,
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
        query = request.form.get("query", "").strip().lower()

        if not query:
            return redirect(url_for("chat"))

        if query in ["hi", "hello", "hey", "hii", "hai"]:
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
        return jsonify({"error": "Empty query"}), 400

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
    app.run(host="0.0.0.0", port=5000, debug=False)
