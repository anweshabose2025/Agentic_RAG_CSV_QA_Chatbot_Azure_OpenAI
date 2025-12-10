# AZURE OPENAI LLM
# streamlit run 2.py

from typing import List
from pydantic import BaseModel
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langgraph.graph import StateGraph, START, END
import tempfile
from langchain_openai import AzureChatOpenAI

# Streamlit config
st.title("🦜: Agentic RAG Chatbot LangGraph")
st.sidebar.title("Azure OpenAI Settings")
st.sidebar.markdown("[Create Azure Openai Resource](portal.azure.com)")
AZURE_OPENAI_API_KEY = st.sidebar.text_input("Openai API Key of Azure resourse", type="password")
AZURE_ENDPOINT_URI = st.sidebar.text_input("Openai Endpoint URI of Azure resourse")
API_VERSION = st.sidebar.text_input("Openai API version")
DEPLOYMENT = st.sidebar.text_input("Openai Deployment name")


if not AZURE_OPENAI_API_KEY or not AZURE_ENDPOINT_URI or not API_VERSION or not DEPLOYMENT:
    st.warning("Please enter proper Azure resource information.")
    st.stop()

else:
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT_URI,
        azure_deployment=DEPLOYMENT,
        api_version=API_VERSION)

# Inputs
file = st.file_uploader("Upload any file that you want the model to learn:")
user_question = st.chat_input("Ask a question:")

if st.button("Learn Document") and file:
    with st.spinner("Learning Document. Please Wait..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file.getbuffer())
            temp_path = tmp.name

        # Load using correct path
        doc = CSVLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        split_docs = splitter.split_documents(doc)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("📚 Document learned successfully! Now you can write your question")


class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""

def retrieve_docs(state: RAGState) -> RAGState:
    docs = st.session_state.retriever.invoke(state.question)
    return RAGState(question=state.question,retrieved_docs=docs)

def generate_answer(state: RAGState) -> RAGState:
    context = "\n\n".join([d.page_content for d in state.retrieved_docs])
    prompt = (
        f"Answer the question based ONLY on the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state.question}")
    response = llm.invoke(prompt)
    return RAGState(
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=response.content)


builder = StateGraph(RAGState)
builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()


if user_question:
    if "retriever" not in st.session_state:
        st.warning("Please click 'Learn Document' first to load the content.")
    else:
        final_state = graph.invoke({"question": user_question})
        st.success(final_state["answer"])