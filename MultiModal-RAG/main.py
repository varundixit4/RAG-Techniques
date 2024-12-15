import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from PIL import Image as PILImage
import io
from base64 import b64decode
import base64
import os
from dotenv import load_dotenv

# Import utility functions
from utils import (
    chunk_pdf,
    separate_elements,
    summarise_texts,
    summarise_images,
    ingest_chunks,
    parse_docs,
    build_prompt,
)

# Load environment variables
load_dotenv()

# Initialize keys and settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Initialize models and vectorstore
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)


# Function to ingest PDF
def ingest_pdf(input_file):
    chunks = chunk_pdf(input_file)
    texts, tables, images = separate_elements(chunks)
    text_summaries, table_summaries = summarise_texts(texts, tables)
    image_summaries = summarise_images(images)

    ingest_chunks(id_key, retriever, texts, tables, images, text_summaries, table_summaries, image_summaries)
    return "PDF successfully ingested. You can now ask questions."


# Function to query and format results
def query(question):
    # Define the chain
    chain = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | model
            | StrOutputParser()
        )
    )

    # Execute the chain
    response = chain.invoke(question)

    # Extract results
    final_response = response['response']
    context_texts = [
    {
        "text": text.text,
        "page_number": getattr(text.metadata, "page_number", "N/A"),
    }
    for text in response['context']['texts']
]
    context_images = response['context']['images']

    return final_response, context_texts, context_images


import gradio as gr
from PIL import Image as PILImage
import base64
import io

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📄 Multimodal RAG Chatbot
        """
    )

    with gr.Row():
        # Left Column: PDF Upload & Ingestion
        with gr.Column(scale=1):
            gr.Markdown("### Upload and Ingest a PDF")
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], interactive=True)
            ingest_output = gr.Textbox(label="Ingestion Status", interactive=False)
            ingest_button = gr.Button("Ingest PDF")

            # Link PDF ingestion
            ingest_button.click(
                fn=ingest_pdf,
                inputs=[pdf_input],
                outputs=[ingest_output],
            )

        # Right Column: Ask Questions (Chatbot Style)
        with gr.Column(scale=2):
            gr.Markdown("### Ask Questions About the PDF")
            
            # Create a Chatbot for the user interaction
            chatbot = gr.Chatbot()
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="What do you want to know about the uploaded PDF?",
                interactive=True,
                show_label=False
            )
            # Collapsible context section
            with gr.Accordion("Context", open=False):
                text_response = gr.Textbox(
                    label="Response",
                    interactive=False,
                    lines=5,
                )

                image_gallery = gr.Gallery(
                    label="Images in Context",
                    elem_id="image-gallery",
                )
            question_button = gr.Button("Submit Question")

            # Query function for Gradio
            def query_and_display(question, history):
                final_response, context_texts, context_images = query(question)

                # Format text response
                context_response = "Context:\n\n"
                for context in context_texts:
                    context_response += f"{context['text']}\n(Page number: {context['page_number']})\n"
                    context_response += "-" * 50 + "\n"

                # Convert base64 image data to PIL Image
                pil_images = []
                for img_base64 in context_images:
                    image_data = base64.b64decode(img_base64)
                    pil_image = PILImage.open(io.BytesIO(image_data))
                    pil_images.append(pil_image)

                # Add the new question-answer pair to the chatbot history
                history.append((question, final_response))

                return history, context_response, pil_images

            # Link query function
            question_button.click(
                fn=query_and_display,
                inputs=[question_input, chatbot],
                outputs=[chatbot, text_response, image_gallery],
            )

# Launch Gradio app
demo.launch()