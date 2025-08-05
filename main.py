import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import for Google's native SDK
import google.generativeai as genai

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Caching Mechanism
retriever_cache = {}

# Mount the 'static' directory to serve files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ApiRequest(BaseModel):
    documents: str
    questions: list[str]

class ApiResponse(BaseModel):
    answers: list[str]

# Core function
async def process_questions(doc_url: str, questions: list[str]) -> list[str]:
    try:
        embedding_api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")
        generative_api_key = os.getenv("GOOGLE_GENERATIVE_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not embedding_api_key or not generative_api_key or not cohere_api_key:
            raise ValueError("All API keys must be set.")

        if doc_url in retriever_cache:
            base_retriever = retriever_cache[doc_url]
        else:
            loader = PyPDFLoader(doc_url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=embedding_api_key)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            retriever_cache[doc_url] = base_retriever

        # Configure the native Google AI client for the generative step
        genai.configure(api_key=generative_api_key)
        llm = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        # Use the Cohere API key for the re-ranker
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=3, cohere_api_key=cohere_api_key)
        
        # Format questions for the batch prompt
        formatted_questions = "\n".join(f"- {q}" for q in questions)
        
        # Retrieve and re-rank the context documents
        initial_docs = await base_retriever.ainvoke(" ".join(questions))
        reranked_docs = await reranker.acompress_documents(documents=initial_docs, query=" ".join(questions))
        context = "\n".join(doc.page_content for doc in reranked_docs)
        
        # Create the final prompt string
        prompt = f"""
            Your task is to answer a list of questions based ONLY on the context provided.
            Provide the answers as a valid JSON object. The JSON object must have a single key called "answers".
            The value of "answers" must be a JSON array of strings, where each string is the answer to a question in the exact same order they were asked.
            Do not include your thought process.

            Here is the list of questions you must answer:
            {formatted_questions}

            <context>
            {context}
            </context>
            """
        
        # --- Single API Call using the native Google SDK ---
        response = await llm.generate_content_async(prompt)

        # Extract and clean the JSON string from the response
        answer_text = response.text.strip().replace("`json", "").replace("`", "")
        
        try:
            answer_json = json.loads(answer_text)
            answers = answer_json.get("answers", ["Error parsing model response." for _ in questions])
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from model response: {answer_text}")
            answers = [f"Error: Model returned invalid JSON." for _ in questions]

        return answers

    except Exception as e:
        print(f"An error occurred: {e}")
        return [f"Error processing document: {e}" for _ in questions]

# API endpoint
@app.post("/hackrx/run", response_model=ApiResponse)
async def hackrx_run(request: Request, body: ApiRequest):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    
    answers = await process_questions(body.documents, body.questions)
    
    return ApiResponse(answers=answers)

# Root endpoint
@app.get("/")
def read_root():
    return {"status": "RAG API is live and ready"}