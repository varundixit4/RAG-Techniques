# Chunk the PDF and Summarise the Chunks
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage
from base64 import b64decode
import uuid
import time

def chunk_pdf(file_path):
    """Chunks down the PDF"""

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            
        strategy="hi_res",

        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,

        chunking_strategy="by_title",      
        max_characters=10000,                 
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        )
    return chunks

def separate_elements(chunks):
    """Segregates the chunks into Text, Tables and Images"""
    tables = []
    texts = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    
    return texts, tables, images

def summarise_texts(texts, tables):
    """Summarise the chunks before storing them in the vector DB"""
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Summarize text in batches
    text_summaries = []
    for text_batch in texts:
        try:
            text_summaries.extend(summarize_chain.batch([text_batch]))
        except Exception as e:
            print(f"Error processing text batch: {e}")
        time.sleep(2)

    # Summarize tables in batches
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = []
    for table_batch in tables_html:
        try:
            table_summaries.extend(summarize_chain.batch([table_batch]))
        except Exception as e:
            print(f"Error processing table batch: {e}")
        time.sleep(2)

    return text_summaries, table_summaries

def summarise_images(images):
    """Summarise the images before storing them in the vector DB"""
    prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    chain = prompt | model | StrOutputParser()

    image_summaries = []
    for image in images:
        image_summary = chain.invoke([image])
        image_summaries.append(image_summary)
        time.sleep(5)

    return image_summaries
    

def ingest_chunks(id_key, retriever, texts, tables, images, text_summaries, table_summaries, image_summaries):
    """Ingest the chunks into document store and summaries into vector store"""

    # Add texts
    if len(texts) > 0:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    if len(tables) > 0:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    if len(images) > 0:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )