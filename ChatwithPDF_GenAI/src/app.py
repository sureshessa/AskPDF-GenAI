from helper import *
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("OPENAI_API_KEY")  
os.environ['OPENAI_API_KEY'] =  OPENAI_API_KEY

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def main():
    docs=get_PDF_data()
    text_chunks=get_textchunks(docs)
    vectordb=get_vector_store(text_chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    '''The RetrievalQA is a chain that combines a Retriever and a QA chain. 
    It is used to retrieve documents from a Retriever and then use a QA chain to answer a question based 
    on the retrieved documents.'''
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    
    while True:
        user_input=input("Ask PDF: (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break;
        else:
            llm_response = qa_chain(user_input)
            process_llm_response(llm_response)


if __name__ == "__main__":
    main()