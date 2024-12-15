from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import (chunk_pdf, 
                   separate_elements, 
                   summarise_chunks, 
                   ingest_chunks, 
                   parse_docs, 
                   build_prompt)
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

input_file = './data/1706.03762v7.pdf'

chunks = chunk_pdf(input_file)
texts, tables, images = separate_elements(chunks)
text_summaries, table_summaries, image_summaries = summarise_chunks(texts, tables, images)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

ingest_chunks(id_key, retriever, texts, tables, images, text_summaries, table_summaries, image_summaries)

chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | model
    | StrOutputParser()
)

response = chain.invoke(
    "What is the attention mechanism?"
)

print(response)