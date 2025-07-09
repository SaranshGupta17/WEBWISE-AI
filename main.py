from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as sl
from urllib.parse import urlparse
import os
from speech_recogniser import mic

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
    url = sl.sidebar.text_input(f"input URL{i+1}")
    if(is_valid_url(url)):
        url_list.append(url)

processed_url = sl.sidebar.button("Process URLs")

session = sl.session_state

if "newq" not in session:
    session.newq = ""

if sl.button("mic"):
    session.newq = mic()           
query = sl.text_input("Write your Question", value = session.newq)


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
if(processed_url):
    loader =  WebBaseLoader(url_list)
    main_placeholder.text("Data Loading Started.")
    chunks = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    main_placeholder.text("Text Splitter Started.")
    document = splitter.split_documents(chunks)
    
    
    vector = FAISS.from_documents(document,embedding)
    main_placeholder.text("Embedding Vector Started Building.")
    
    file_path = "vector_index"
    vector.save_local(file_path)
    main_placeholder.text("Embedding completed.")


if query:
        if os.path.exists(file_path):
            
            vector = FAISS.load_local(file_path,embedding,allow_dangerous_deserialization=True)
            qa_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=custom_prompt,document_variable_name="context")
            chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain,retriever = vector.as_retriever())
            ans = chain({"question":query})
            
            print(ans)
            sl.header("answer")
            sl.subheader(ans["answer"])       
            sl.header("Sources:")
            sources = ans["sources"].split("\n")
            for i in sources:
                sl.write(i)
                
            session.newq = ""

    