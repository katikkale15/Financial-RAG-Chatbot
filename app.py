import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load DB
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="financial_pdfs")
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_retriever()
llm = OllamaLLM(model="llama3.2")

# Prompt
prompt = ChatPromptTemplate.from_template(
    """Financial analyst. Answer from context only.
    Context: {context}
    Question: {input}
    Answer:"""
)

# Simple RAG Chain (runnables only)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit
st.title("💰 Financial RAG Chatbot")
st.info("Ollama + ChromaDB powered.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about PDFs:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    with st.chat_message("assistant"):
        response = rag_chain.invoke(query)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
