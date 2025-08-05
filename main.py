import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
# --- Import for Re-ranking ---
from langchain_cohere import CohereRerank
from langchain.chains.retrieval import create_retrieval_chain

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
            raise ValueError("All API keys (GOOGLE_EMBEDDING, GOOGLE_GENERATIVE, COHERE) must be set.")

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

        # Use the second API key for the chat model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=generative_api_key)
        
        # Use the third API key for the re-ranker
        reranker = CohereRerank(top_n=3, cohere_api_key=cohere_api_key)
        
        # Batch Prompt for a single API call
        formatted_questions = "\n".join(f"- {q}" for q in questions)
        
        prompt = ChatPromptTemplate.from_template(
            """
            Your task is to answer a list of questions based ONLY on the context provided.
            Provide the answers as a valid JSON object. The JSON object must have a single key called "answers".
            The value of "answers" must be a JSON array of strings, where each string is the answer to a question in the exact same order they were asked.
            Do not include your thought process.

            Here is the list of questions you must answer:
            {input}

            <context>
            {context}
            </context>
            """
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create a new retrieval chain that includes the re-ranker
        # Note: The re-ranker is not directly part of the LCEL chain in this batch setup.
        # We manually retrieve, re-rank, and then invoke the document chain.

        # Retrieve initial context based on all questions
        initial_docs = await base_retriever.ainvoke(" ".join(questions))
        
        # Re-rank the retrieved documents
        reranked_docs = await reranker.acompress_documents(documents=initial_docs, query=" ".join(questions))
        
        # Single API Call with re-ranked context
        response = await document_chain.ainvoke({
            "input": formatted_questions,
            "context": reranked_docs
        })

        # The model's response should be a JSON string containing the list of answers
        answer_text = response.strip()
        
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