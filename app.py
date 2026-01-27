from flask import Flask, request, render_template, session, redirect, url_for
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os

# ============================================================
# APP CONFIG
# ============================================================

app = Flask(__name__)
app.secret_key = "super-secret-key-123"

# ============================================================
# LOAD WEBSITE CONTENT (RAG SOURCE)
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
# LOCAL LLM (OLLAMA)
# ============================================================

llm = OllamaLLM(
    model="llama3",
    temperature=0.2
)

# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant for SkyNet Fiber.

Answer ONLY using the context.
Use bullet points if needed.
If the answer is not found, clearly say you do not have that information.

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
# ROUTES
# ============================================================

# 🔹 HOME PAGE (ISP WEBSITE)
# Opens FIRST when app.py is executed
@app.route("/")
def home():
    return render_template("home.html")


# 🔹 CHATBOT PAGE
# Opens when Chat button is clicked
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        query = request.form.get("query", "").strip().lower()

        if not query:
            return redirect(url_for("chat"))

        # Greeting shortcut
        if query in ["hi", "hello", "hey", "hii", "hai"]:
            answer = "Hello 👋 How can I help you today?\n\nAre your doubts clarified?"
        else:
            docs = retriever.invoke(query)

            if not docs:
                answer = (
                    "I can answer only SkyNet Fiber related queries.\n\n"
                    "Are your doubts clarified?"
                )
            else:
                context = "\n\n".join(d.page_content for d in docs)
                answer = (
                    llm.invoke(
                        prompt.format(
                            context=context,
                            question=query
                        )
                    ).strip()
                    + "\n\nAre your doubts clarified?"
                )

        session["chat"].append(("user", query))
        session["chat"].append(("bot", answer))
        session.modified = True

    return render_template("index.html", chat=session["chat"])


# 🔹 CLEAR CHAT
@app.route("/clear")
def clear_chat():
    session.pop("chat", None)
    return redirect(url_for("chat"))


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
