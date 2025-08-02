import os
import uvicorn
import json
from fastapi import FastAPI, Request, HTTPException
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

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
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        if doc_url in retriever_cache:
            retriever = retriever_cache[doc_url]
        else:
            loader = PyPDFLoader(doc_url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            retriever = vectorstore.as_retriever()
            retriever_cache[doc_url] = retriever

        llm = ChatGroq(
            temperature=0.1,
            model="gemma2-9b-it",
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        
        # Final prompt with escaped curly braces for the example
        prompt = ChatPromptTemplate.from_template(
            """
            Your task is to answer the question based ONLY on the context provided.
            Your final response MUST be a valid JSON object.
            It must contain a single key called "answer".
            The value of the "answer" key MUST be a single, natural-language sentence correctly enclosed in double quotes.

            For example:
            {{
                "answer": "This is a correct, natural language answer in a valid JSON string."
            }}

            Do NOT just write the text. It MUST be a valid JSON string value.

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
            
            answers.append(str(final_answer))

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