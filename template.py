import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
from langchain.tools.retriever import create_retriever_tool

import google.generativeai as genai # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv 

from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit app
st.title("PDF Q&A with Langchain and Google Generative AI")

def get_pdf_text(file_paths):
    """
    Load PDF documents from the provided file paths.
    """
    all_docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # print(docs)
        all_docs.extend(docs)
        # print(type(all_docs))
    return all_docs

def get_text_chunks(docs):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)
    # print(docs)
    return documents

def get_vector_store(documents):
    """
    Create a vector store using Google Generative AI Embeddings and FAISS.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    # print(embeddings)
    # print(vector_store)
    retriever = vector_store 
    
    return retriever

    retriever_tool=create_retriever_tool(retriever,"documents",
                      "Search for information about document. For any questions about your query, you must use this tool!")
    
    print(retriever_tool.name)

def get_conversational_chain():
    """
    Create a conversational chain for question-answering.
    """
    # Define the prompt template for question-answering
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if the user finds the answer helpful.
    <context>
    {context}
    </context>
    Question: {input}
    """)
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )
    # Load a question-answering chain using the language model and prompt
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain = create_stuff_documents_chain(model, prompt)
    return document_chain

# Main function
def main():
    # Sidebar file uploader for PDF files
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        file_paths = [file.name for file in uploaded_files]  # Get the file paths of uploaded files
        
        st.write("Files uploaded successfully!")

        # Load and process all PDF files
        all_docs = get_pdf_text(file_paths)
        
        # Split text into chunks
        text_chunks = get_text_chunks(all_docs)
        
        # Create vector store
        vector_store = get_vector_store(text_chunks)
        
        st.write("Data processing complete! You can now ask questions.")
        

        # Input for user question
        user_question = st.text_input("Ask your question:")
        
        if user_question:
            # Search for similar documents in the vector store
            retriever = vector_store.as_retriever()
            from langchain.chains import create_retrieval_chain
            retrieval_chain = create_retrieval_chain(retriever, get_conversational_chain())
            response = retrieval_chain.invoke({"input": user_question})
            
            # Display the answer
            st.write("Answer: ", response["answer"])

# Run the app
if __name__ == "__main__":
    main()




