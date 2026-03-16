import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_rag_chain(retriever):
    """Creates a RAG chain for question answering."""
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def handle_query(query, vector_db):
    """Orchestrates the retrieval and generation process."""
    if not vector_db:
        return "No documents uploaded yet. Please upload a document first."
    
    retriever = vector_db.as_retriever()
    chain = get_rag_chain(retriever)
    
    response = chain.invoke({"input": query})
    return response["answer"], response["context"]
