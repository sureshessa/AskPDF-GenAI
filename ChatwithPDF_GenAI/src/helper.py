import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


#Load pdf documents from data folder
def get_PDF_data():
    loader=DirectoryLoader('./data',
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents


#Split Text into Chunks
def get_textchunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=1000,
                                                chunk_overlap=200)
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks

#Convert text chuncks into embeddings and store them into Chroma db
def get_vector_store(text_chunks):
    persist_directory = 'db'

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(documents=text_chunks,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
    return vectordb

