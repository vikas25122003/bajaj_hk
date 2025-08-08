import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Import Groq's async client ---
from groq import AsyncGroq

# LangChain Imports for the RAG pipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not embedding_api_key or not groq_api_key or not cohere_api_key:
            raise ValueError("All API keys (GOOGLE_EMBEDDING, GROQ, COHERE) must be set.")

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

        # --- Initialize the native Groq Async Client ---
        groq_client = AsyncGroq(api_key=groq_api_key)
        
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=3, cohere_api_key=cohere_api_key)
        
        formatted_questions = "\\n".join(f"- {q}" for q in questions)
        
        initial_docs = await base_retriever.ainvoke(" ".join(questions))
        reranked_docs = await reranker.acompress_documents(documents=initial_docs, query=" ".join(questions))
        context = "\\n".join(doc.page_content for doc in reranked_docs)
        
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
        
        # --- Single API Call using the native Groq SDK ---
        completion = await groq_client.chat.completions.create(
            model="openai/gpt-oss-20b", # Using a reliable model instead of the one from the snippet
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_tokens=8192,
            top_p=1,
            stream=False, # We need the full response, so stream is False
        )

        answer_text = completion.choices[0].message.content
        
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