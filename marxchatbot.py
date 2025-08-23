from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
import streamlit as st
import os

DATA_DIR = "/Users/nnaumova/Desktop/Data Science Course/Projects/comchatbot/data"              # data directory
PERSIST_DIR = "/Users/nnaumova/Desktop/Data Science Course/Projects/comchatbot/vector_index"   # directiry for index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    cache_folder="/Users/nnaumova/Desktop/Data Science Course/Projects/comchatbot/embedding_model"
)

def get_or_create_index():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embeddings)
    else:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter],
            embed_model=embeddings
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

input_template = """Here is the context:
{context_str} 
You are a Marxist assistant.
Greet user when they say hi or hello and ask, wheather they would like to explore Marx' ideas.
The user does not know anything about Marxism. 
If the user replies with "yes" or "no", interpret it as a response to your last question and continue talking about the same topic.
Ground your proposed topic for discussion and your answers in Karl Marx's *Capital* if possible.
Do NOT invent random context unless it is directly relevant to the question. 
Always continue the conversation naturally.
Do NOT comment on the type of question or the user's response. 
Do NOT give personal opinions or meta-comments. 
Keep answers concise and clear.
Question: {query_str}
Answer:"""

prompt = PromptTemplate(template=input_template)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Marxist chatbot")

st.write("Get answers to your questions about the economy and society from the three volumes of Capital.")

# Loading index
index = get_or_create_index()
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
chat_engine = index.as_chat_engine(chat_mode="context", llm=llm, memory=memory, verbose=True)

# UI for questions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if user_input:

    retriever = index.as_retriever(similarity_top_k=3)
    if user_input.lower().strip() in ["yes", "no", "да", "нет"]:
        context_text = ""
    else:
        context_nodes = retriever.retrieve(user_input)
        context_text = "\n".join([node.get_text() for node in context_nodes])

    filled_prompt = prompt.format(
    context_str=context_text,
    query_str=user_input)

    response = llm.complete(filled_prompt)
    answer_text = response.text
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Marxist-bot", answer_text))

for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")