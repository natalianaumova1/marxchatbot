**Marxist Chatbot (Streamlit + LlamaIndex)**

An interactive chatbot that answers questions about economics, politics, and society, drawing on Karl Marx’s Capital.
Built with Streamlit and powered by LlamaIndex for retrieval-augmented generation (RAG).

**Features**

Answers questions based on relevant passages from uploaded documents.

Retrieval-Augmented Generation (RAG) with Hugging Face embeddings.

Persists local embeddings and vector index across runs.

Runs both locally and in the cloud (Streamlit Cloud).

**Project Structure**
.
├── data/                 # documents (PDF, TXT, etc.)
├── embedding_model/      # cache for embeddings (auto-created)
├── vector_index/         # saved index (auto-created)
├── marxchatbot.py        # main Streamlit app
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
