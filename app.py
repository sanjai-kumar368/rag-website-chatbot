from flask import Flask, request, render_template, session, redirect, url_for
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = "super-secret-key-123"

# -------------------- Load Website Content --------------------
with open("data/website_content.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------- Embeddings --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- Local LLM (Ollama) --------------------
llm = OllamaLLM(
    model="llama3",
    temperature=0.2
)

# -------------------- Prompt Template --------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant for SkyNet Fiber.

Answer the user's question ONLY using the information provided in the context.
Keep the answer concise and well-structured.
If listing services, features, or plans, use bullet points.
Each bullet must be on a new line.
Do not add any information that is not in the context.
If the answer is not present in the context, say you do not have that information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# -------------------- FAISS Vector Store --------------------
FAISS_PATH = "faiss_index"
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

if os.path.exists(FAISS_PATH):
    print("🔁 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("⚙️ Creating FAISS index...")
    docs = text_splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print("✅ FAISS index saved")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        query = request.form["query"].strip()
        query_lower = query.lower()

        # -------- YES / NO HANDLING --------
        if query_lower in ["yes", "yeah", "yep", "yes it is", "clarified"]:
            answer = "Glad to hear that! 😊 If you have any more questions about SkyNet Fiber, feel free to ask."

        elif query_lower in ["no", "not yet", "nope", "nah"]:
            answer = (
                "I'm sorry about that. 🙏\n\n"
                "You can contact SkyNet Fiber customer support directly:\n\n"
                "📞 Toll-free: 1800-890-2020\n"
                "📞 Alternate: +91 44 4200 8899\n"
                "📧 Email: support@skynetfiber.in\n\n"
                "Our support team is available 24/7."
            )

        # -------- GREETINGS --------
        elif query_lower in ["hi", "hello", "hey", "hii", "hai"]:
            answer = "Hello 👋 I can help you with SkyNet Fiber plans, support, and services.\n\nAre your doubts clarified?"

        # -------- NORMAL RAG FLOW --------
        else:
            docs = retriever.invoke(query)

            if not docs:
                answer = (
                    "I can answer only questions related to SkyNet Fiber services.\n\n"
                    "Are your doubts clarified?"
                )
            else:
                context = "\n\n".join([doc.page_content for doc in docs])

                answer = llm.invoke(
                    prompt.format(
                        context=context,
                        question=query
                    )
                )

                # 🔹 FOLLOW-UP QUESTION (ALWAYS)
                answer = answer.strip() + "\n\nAre your doubts clarified?"

        session["chat"].append(("user", query))
        session["chat"].append(("bot", answer))
        session.modified = True

    return render_template("index.html", chat=session["chat"])


@app.route("/clear")
def clear_chat():
    session.pop("chat", None)
    return redirect(url_for("home"))

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
