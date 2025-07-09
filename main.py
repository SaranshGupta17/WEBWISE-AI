from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import RetrievalQAWithSourcesChain
from langchain.chains.llm import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as sl
from urllib.parse import urlparse
import os
from speech_recogniser import mic

# Optional: Prevent WebBaseLoader warning
os.environ["USER_AGENT"] = "WebWiseAI/1.0"

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
file_path = ""

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return all([parsed.scheme in ["http", "https"], parsed.netloc])
    except Exception:
        return False

sl.title("News Research Tool")
sl.sidebar.title("News Article URLs")

url_list = []
for i in range(6):
    url = sl.sidebar.text_input(f"Input URL {i+1}")
    if is_valid_url(url):
        url_list.append(url)

processed_url = sl.sidebar.button("Process URLs")

# Initialize session state
session = sl.session_state
if "newq" not in session:
    session.newq = ""

# Mic input
if sl.button("🎤 Mic"):
    session.newq = mic()

# Question input
query = sl.text_input("Write your Question", value=session.newq)

# Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert researcher assistant.

Use the following context to answer the question as comprehensively and clearly as possible. 
If the answer exists in multiple sources, combine them into a well-structured explanation. 
Include citations (like [Source 1]) at the end of each fact. 
Only answer from the context below. Do not make up anything.

==========
{context}
==========

Question: {question}
Answer:
"""
)

main_placeholder = sl.empty()

# Process URLs
if processed_url:
    loader = WebBaseLoader(url_list)
    main_placeholder.text("🔄 Data Loading Started...")
    chunks = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    main_placeholder.text("🔄 Text Splitting...")
    documents = splitter.split_documents(chunks)

    vector = FAISS.from_documents(documents, embedding)
    main_placeholder.text("🔄 Building Embedding Vector...")

    file_path = "vector_index"
    vector.save_local(file_path)
    main_placeholder.text("✅ Embedding Completed!")

# Answer Query
if query:
    if os.path.exists(file_path):
        vector = FAISS.load_local(file_path, embedding, allow_dangerous_deserialization=True)

        # Build chain 
        llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
        chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=stuff_chain,
            retriever=vector.as_retriever()
        )

        ans = chain.invoke({"question": query})

        sl.header("Answer")
        sl.subheader(ans["answer"])

        sl.header("Sources:")
        for source in ans["sources"].split("\n"):
            sl.write(source)

        # ✅ Clear the mic input after result
        session.newq = ""
