import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def read_pdf_text(documents):
    """
    This code reads all the pdf documents and returns a combined text
    :param documents: List of pdfs passed
    :return str:Extracted text from all the pdf
    """
    text = ""
    for document in documents:
        reader = PdfReader(document)
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_chunks(text):
    """
    This code will split the data into chunks of 10000 words and have an overlap of 1000
    words, so that if there are any imp values around the boundatry, they dont get missed.
    :param text: text containing all the pdf texts
    :return: chunks: return a list of splitted texts
    """
    splitter_obj = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter_obj.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """creates a vector store in local """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    #faiss_index will be the folder
    vector_store.save_local("faiss_index")


def conversation_chain():
    """creates a conversation chain using the prompt template"""
    prompt = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def input_from_user(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_database = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_database.similarity_search(question)

    chain = conversation_chain()

    response = chain({"input_documents":docs, "question":question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with your PDFs\n(powered by Google Gemini Pro)")

    user_question = st.text_input("What would you like to ask?")

    if user_question:
        input_from_user(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Submit", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Running..."):
                raw_text = read_pdf_text(pdf_docs)
                text_chunks = split_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Completed")


if __name__ == '__main__':
    main()

