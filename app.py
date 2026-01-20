from flask import Flask, request
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# Load text
with open("data/website_content.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([text])

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        docs = retriever.invoke(query)
        answer = docs[0].page_content if docs else "No answer found."

    return f"""
    <h2>RAG Website Chatbot</h2>
    <form method="post">
        <input name="query" style="width:300px">
        <button type="submit">Ask</button>
    </form>
    <p><b>Answer:</b> {answer}</p>
    """

if __name__ == "__main__":
    app.run(debug=True)
