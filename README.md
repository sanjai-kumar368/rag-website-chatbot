<img width="1897" height="860" alt="image" src="https://github.com/user-attachments/assets/5d00bb34-5d65-4135-b215-160ed6003172" /># SkyNet Fiber – RAG-Powered Customer Support Chatbot

SkyNet Fiber is a full-stack web application that simulates a real-world Internet Service Provider (ISP) website integrated with an AI-powered customer support chatbot.  
The chatbot uses **Retrieval-Augmented Generation (RAG)** to answer user queries strictly based on curated company content.

This project demonstrates the practical application of **Large Language Models (LLMs), vector databases, and web technologies** to build an accurate and scalable customer support system.
This currently hosted on http://13.60.78.170:5000/

---

## Project Overview

The application consists of:
- A responsive ISP website frontend
- A backend Flask server
- An AI-powered chatbot embedded in the website
- Retrieval-Augmented Generation to ensure accurate answers
- Cloud-based LLM integration using Groq

The chatbot retrieves relevant information from indexed company documentation and generates responses without hallucination.

---

## Key Features

- Interactive ISP website with service plans and details
- Floating, draggable, and resizable chatbot interface
- Retrieval-Augmented Generation (RAG) architecture
- Vector similarity search using FAISS
- Context-aware answers restricted to company data
- Secure API key handling using environment variables
- Free-tier compatible deployment

---

## Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask

### AI / NLP
- LangChain
- FAISS (Vector Database)
- Sentence Transformers (Embeddings)
- Groq LLM API

### Deployment
- GitHub
- Render (Free tier)

---

## System Architecture

1. Company content is stored in a structured text file.
2. Content is split into chunks using LangChain text splitters.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in a FAISS vector database.
5. User queries are converted into embeddings.
6. Relevant context is retrieved using similarity search.
7. The retrieved context is passed to the Groq LLM.
8. The chatbot generates an answer strictly based on the retrieved content.

---

## Retrieval-Augmented Generation (RAG)

This project follows a strict RAG approach to ensure reliability:
- The chatbot answers only using provided company content
- No hallucinated or external information is generated
- If information is unavailable, the chatbot clearly states the limitation
- Suitable for real-world customer support applications

---

## Project Structure

backend/
│
├── app.py
├── requirements.txt
├── data/
│ └── website_content.txt
├── templates/
│ ├── home.html
│ └── index.html
├── static/
│ ├── home.css
│ ├── style.css
│ └── images/
└── faiss_index/


---

## Setup Instructions (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/sanjai-kumar368/rag-website-chatbot.git
cd rag-website-chatbot/backend
```
### 2. Create Virtual Environment

Activate the virtual environment:

- Windows:
 ```bash
venv\Scripts\activate
```

- Linux / macOS:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Set the Groq API key:

- Windows (PowerShell):
```bash
$env:GROQ_API_KEY="your_groq_api_key"
```

- Linux / macOS:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at:
```bash
http://127.0.0.1:5000/
```

---

## Deployment

The application is designed to be deployed on **free cloud platforms** such as AWS.

Deployment highlights:
- No local LLM required
- Uses cloud-based Groq LLM
- Environment variables used for security
- Compatible with free-tier memory limits

---

## Use Cases

- ISP customer support automation
- AI-powered FAQ systems
- College and academic projects
- Portfolio demonstration of RAG systems
- Enterprise knowledge-based chatbots

---

## Future Enhancements

- User authentication
- Chat history persistence
- Streaming responses
- Analytics dashboard
- Multi-language support

##Screen shots
<img width="1896" height="858" alt="image" src="https://github.com/user-attachments/assets/bdd28543-0d95-48c9-9877-38200eac7bd1" />
<img width="1898" height="856" alt="image" src="https://github.com/user-attachments/assets/7b0f643a-7486-4db8-8c43-7405f9574704" />
<img width="1892" height="857" alt="image" src="https://github.com/user-attachments/assets/df8abcc1-b412-4b08-a625-21eb91e30acc" />
<img width="1897" height="860" alt="image" src="https://github.com/user-attachments/assets/6dd2238c-ebad-4f27-b156-c3ca39d94cae" />
<img width="1897" height="859" alt="image" src="https://github.com/user-attachments/assets/a1ebe280-4d6d-4558-aa73-53125ddf4e1d" />




