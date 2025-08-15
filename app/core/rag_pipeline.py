from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from services.document_processor import vector_store
from .config import settings

def get_rag_chain(return_retriever: bool = False):
    """Initializes and returns the RAG chain, and optionally the retriever."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY
    )

    # Get the retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if return_retriever:
        return rag_chain, retriever
    return rag_chain
