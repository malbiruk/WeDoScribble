import base64
import json
from pathlib import Path

import requests
from langchain.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.unstructured import \
    UnstructuredFileLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from tools.web_search import summarize

llm = ChatOllama(model="llava:7b", temperature=0)


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


@tool
def see_image(file_path: str, query: str) -> str:
    """
    fetches and returns the text content \
    of a single image file, which path is provided to "file_path".
    "query" is a question about the image.
    """
    with Path(file_path).open("rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    chain = prompt_func | llm | StrOutputParser()

    return chain.invoke(
        {"text": query, "image": image_b64},
    )


@tool
def read_pdf(file_path: str, query: str) -> str:
    """
    fetches and returns the summarized text content \
    of a single .pdf file, which path is provided to "file_path".
    "query" should be specified in order to create summary,
    it is an objective of reading the pdf
    """
    loader = PyMuPDFLoader(Path(file_path))
    docs = loader.load()
    data = "".join([doc.page_content for doc in docs])
    if len(data) > 85000 * 4:
        return summarize(query, data)
    return docs


@tool
def read_any_file(file_path: str, query: str) -> str:
    """
    fetches and returns the summarized text content \
    of a single file (a lot of formats are supported: tables, html, documents),
    which path is provided to "file_path".
    "query" should be specified in order to create summary,
    it is an objective of reading the file
    """
    loader = UnstructuredFileLoader(Path(file_path), mode="single")
    data = loader.load()
    if len(data[0].page_content) > 85000 * 4:
        return summarize(query, data[0].page_content)
    return data


@tool
def read_google_docs(url: str) -> str:
    """
    use this tool if you need to read contents of the docs.google.com link

    it downloads the content of the google doc to temporary location, \
    and you'll need to read contents of the downloaded file using read_any_file tool
    """
    splitted_url = url[url.find("docs.google.com"):].split("/")
    doc_type = splitted_url[1]
    doc_id = splitted_url[3]

    doc_type_to_format = {"document": "docx",
                          "spreadsheets": "xlsx",
                          "presentation": "pptx"}

    format_ = doc_type_to_format.get(doc_type)

    if not format_:
        return "Couldn't process google doc -- invalid URL."

    download_url = f"https://docs.google.com/{doc_type}/d/{doc_id}/export?format={format_}"
    file_path = Path(f"tmp.{format_}")
    response = requests.get(download_url, timeout=10)
    if response.status_code == 200:
        with Path(file_path).open("wb") as file:
            file.write(response.content)
        return (f"File downloaded successfully to {file_path.absolute()}. "
                "Now read it using read_any_file tool.")
    return "Failed to download the file."
