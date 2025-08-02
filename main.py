import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
# --- Use Google's Gemini for Embeddings ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Mount the 'static' directory to serve files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for API request and response structure
class ApiRequest(BaseModel):
    documents: str
    questions: list[str]

class ApiResponse(BaseModel):
    answers: list[str]

# Core function to process documents and questions
async def process_questions(doc_url: str, questions: list[str]) -> list[str]:
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        loader = PyPDFLoader(doc_url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # --- Use Google's model for embeddings ---
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatGroq(
            temperature=0.1,
            model="deepseek-r1-distill-llama-70b",
            model_kwargs={"response_format": {"type": "json_object"}},
        )

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

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        answers = []
        for question in questions:
            response = await retrieval_chain.ainvoke({"input": question})
            answer_content = response.get("answer", '{}')
            
            try:
                answer_json = json.loads(answer_content)
                final_answer = answer_json.get("answer", "Could not find an answer in the expected format.")
            except json.JSONDecodeError:
                final_answer = f"Error: Model did not return valid JSON. Received: {answer_content}"
            
            # This line fixes the Pydantic validation error
            answers.append(str(final_answer))

        return answers

    except Exception as e:
        print(f"An error occurred: {e}")
        return [f"Error processing document: {e}" for _ in questions]

# API endpoint required by the hackathon
@app.post("/hackrx/run", response_model=ApiResponse)
async def hackrx_run(request: Request, body: ApiRequest):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    
    answers = await process_questions(body.documents, body.questions)
    
    return ApiResponse(answers=answers)

# Root endpoint for basic "API is live" check
@app.get("/")
def read_root():
    return {"status": "RAG API is live and ready"}