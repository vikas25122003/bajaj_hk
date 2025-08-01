import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import necessary libraries for LangChain, Groq, and OpenAI
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (GROQ_API_KEY, OPENAI_API_KEY)
load_dotenv()

# --- 1. Define the API and data models ---
app = FastAPI()

class ApiRequest(BaseModel):
    documents: str
    questions: list[str]

class ApiResponse(BaseModel):
    answers: list[str]

# --- 2. Create the core RAG function ---
async def process_questions(doc_url: str, questions: list[str]) -> list[str]:
    """
    This function processes a PDF from a URL and answers questions using Groq and OpenAI embeddings.
    """
    try:
        # Load the document from the URL
        loader = PyPDFLoader(doc_url)
        docs = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings()
        
        # Create the vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Define the Groq LLM with JSON mode enabled
        llm = ChatGroq(
            temperature=0.1,
            model="llama3-8b-8192", # Using a valid, high-performance Groq model
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        # Create a prompt template with an explicit JSON instruction
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context.
            Provide a concise and direct answer. Your response MUST be a JSON object
            with a single key called "answer".

            <context>
            {context}
            </context>

            Question: {input}
            """
        )

        # Create the processing chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        answers = []
        for question in questions:
            response = await retrieval_chain.ainvoke({"input": question})
            answer_content = response.get("answer", '{}')
            
            try:
                # The model's output is a JSON string, so we parse it
                answer_json = json.loads(answer_content)
                final_answer = answer_json.get("answer", "Could not find an answer in the expected format.")
            except json.JSONDecodeError:
                final_answer = f"Error: Model did not return valid JSON. Received: {answer_content}"
            
            answers.append(final_answer)

        return answers

    except Exception as e:
        print(f"An error occurred: {e}")
        return [f"Error processing document: {e}" for _ in questions]


# --- 3. Define the API Endpoint for the Hackathon ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def hackrx_run(request: Request, body: ApiRequest):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    
    answers = await process_questions(body.documents, body.questions)
    
    return ApiResponse(answers=answers)


# --- 4. Add a root endpoint for basic testing ---
@app.get("/")
def read_root():
    return {"status": "Groq RAG API with OpenAI embeddings is live"}