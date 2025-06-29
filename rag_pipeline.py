import os
import logging
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

import torch

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Model & Embedding Initialization

# Load sentence-transformer embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local LLM setup using Hugging Face transformers
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Detect if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Define Hugging Face pipeline
text2text_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Wrap with LangChain's interface
llm = HuggingFacePipeline(pipeline=text2text_pipe)
llm.pipeline.model.config.temperature = 0

# Document Handling Functions

def load_and_split_documents(pdf_path: str) -> list:
    """Loads and splits a PDF into document chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_store(pdf_path: str = "documents/sample.pdf", store_dir: str = "vector_store") -> None:
    """Builds and saves a FAISS vector store from a PDF."""
    logging.info(f"Building vector store from: {pdf_path}")
    documents = load_and_split_documents(pdf_path)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(store_dir)
    logging.info(f"Vector store saved to: {store_dir}")

def get_answer(question: str, use_custom: bool = False, custom_path: str = None) -> str:
    logging.info(f"Received question: {question}, use_custom={use_custom}, custom_path={custom_path}")
    try:
        if use_custom and custom_path:
            if not os.path.exists(custom_path):
                logging.error(f"Custom file not found: {custom_path}")
                return "The provided PDF file does not exist."
            logging.info(f"Using custom document: {custom_path}")
            documents = load_and_split_documents(custom_path)
            logging.info(f"Loaded {len(documents)} document chunks from custom PDF")
            db = FAISS.from_documents(documents, embeddings)
        else:
            store_path = "vector_store"
            logging.info(f"Using default vector store at: {store_path}")
            if not os.path.exists(f"{store_path}/index.faiss"):
                logging.warning("Vector store not found. Building now...")
                build_vector_store()
            db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

        retriever = db.as_retriever(search_type="mmr",search_kwargs={"k": 2})
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                Answer the question briefly . No explanations.
                If the answer is not in the context, say "Answer not found."
                Do not include any additional information or explanations.
                Do not inculde unrelated content.
                Answer in the context only.
                Use the following pieces of context to answer the question.

                Context:
                {context}

                Question:
                {question}

                Brief Answer:"""
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )

        result = qa_chain.invoke({"query": question})  # pass question as dict

        return result["result"] if isinstance(result, dict) and "result" in result else str(result)

    except Exception as e:
        logging.error("Error in get_answer", exc_info=True)
        return "Sorry, something went wrong while processing your request."
