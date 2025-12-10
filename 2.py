# Azure Openai
# ---------------------------------------------
#  AGENTIC RAG + DATAFRAME ANALYSIS WITH LANGGRAPH
# ---------------------------------------------

from typing import List
from pydantic import BaseModel
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("🦜: Agentic RAG + DataFrame Analytics (LangGraph)")
st.sidebar.title("Azure OpenAI Settings")

AZURE_OPENAI_API_KEY = st.sidebar.text_input("API Key", type="password")
AZURE_ENDPOINT_URI = st.sidebar.text_input("Azure Endpoint")
API_VERSION = st.sidebar.text_input("API Version")
DEPLOYMENT = st.sidebar.text_input("Deployment Name")

if not (AZURE_OPENAI_API_KEY and AZURE_ENDPOINT_URI and API_VERSION and DEPLOYMENT):
    st.warning("⚠ Please enter all Azure resource info.")
    st.stop()

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_URI,
    api_version=API_VERSION,
    azure_deployment=DEPLOYMENT,
    temperature=0.0
)

file = st.file_uploader("Upload a CSV file to train & analyze:")
user_question = st.chat_input("Ask a question about the file:")


# =====================================================
# LOAD CSV
# =====================================================
if st.button("Learn Document") and file:
    with st.spinner("Reading and embedding document..."):
        with open(f"_{file.name}", "wb") as f:
            f.write(file.read())
        docs = CSVLoader(f"_{file.name}").load()
        st.session_state.df = pd.read_csv(f"_{file.name}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever() #search_type="similarity",search_kwargs={"k": 5})

    st.success("📚 Document learned successfully! Now you can ask questions.")

# =====================================================
# LANGGRAPH STATE
# =====================================================
class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""


# =====================================================
# 🧠 Router: Determine if question needs DataFrame or RAG
# =====================================================
def is_data_query(q: str) -> bool:
    q = q.lower()
    keywords = [
        "sum", "total", "average", "avg", "mean", "median", "filter", "greater than",
        "less than", "top", "bottom", "min", "max", "count", "how many", "rows",
        "columns", "group", "sort", "unique", "distinct"
    ]
    return any(k in q for k in keywords)


def router_node(state: RAGState) -> RAGState:
    """Route to DataFrame node if question is computational."""
    if is_data_query(state.question):
        state.answer = "DATA_QUERY"
    else:
        state.answer = "RAG_QUERY"
    return state


# =====================================================
# 🧮  DATAFRAME ANALYTICS NODE
# =====================================================
def dataframe_node(state: RAGState) -> RAGState:
    df = st.session_state.df
    question = state.question

    # Convert question → Python pandas code
    instruction = f"""
You are a Python data analyst.

Write ONLY Python code using the pandas DataFrame "{df}"
to answer this question: "{question}"

Rules:
- Do NOT assign df again.
- Do NOT print anything.
- The final line must be the return value.
- Code must return a numeric or textual result.
"""
    python_code = llm.invoke(instruction).content.strip()

    #try:
    result = eval(python_code)
    return RAGState(question=question, answer=str(result))
    #except Exception as e:
    #    return RAGState(question=question, answer=f"Error executing code:\n{e}\n\nCode:\n{python_code}")


# =====================================================
# 📚 RETRIEVER NODE
# =====================================================
def retriever_node(state: RAGState) -> RAGState:
    retriever = st.session_state.retriever
    docs = retriever.invoke(state.question)
    return RAGState(question=state.question, retrieved_docs=docs)


# =====================================================
# ✍️ GENERATE ANSWER FROM RETRIEVED DOCS
# =====================================================
def rag_answer_node(state: RAGState) -> RAGState:
    context = "\n\n".join([d.page_content for d in state.retrieved_docs])

    prompt = f"""
You are a helpful assistant. Answer using ONLY the context below.
If answer is not present, say "Not found in document."

CONTEXT:
{context}

QUESTION:
{state.question}
"""
    response = llm.invoke(prompt).content
    return RAGState(question=state.question, retrieved_docs=state.retrieved_docs, answer=response)


# =====================================================
# BUILD LANGGRAPH
# =====================================================
graph = StateGraph(RAGState)

graph.add_node("router", router_node)
graph.add_node("dataframe", dataframe_node)
graph.add_node("retriever", retriever_node)
graph.add_node("rag_answer", rag_answer_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda state: state.answer,  # router sets .answer = DATA_QUERY or RAG_QUERY
    {
        "DATA_QUERY": "dataframe",
        "RAG_QUERY": "retriever"
    }
)
graph.add_edge("retriever", "rag_answer")
graph.add_edge("dataframe", END)
graph.add_edge("rag_answer", END)

graph = graph.compile()


# =====================================================
# RUN QUERY
# =====================================================
if user_question:
    if "retriever" not in st.session_state:
        st.warning("Upload a CSV and click Learn Document first.")
    else:
        final_state = graph.invoke({"question": user_question})
        st.write("Your Question: ", user_question)
        st.success(final_state["answer"])
