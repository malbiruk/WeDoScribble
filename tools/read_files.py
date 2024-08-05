from pathlib import Path

import requests
from langchain.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.unstructured import \
    UnstructuredFileLoader
from tools.google_search import summarize


@tool
def read_pdf(file_path: str, query: str) -> str:
    '''
    fetches and returns the summarized text content \
    of a single .pdf file, which path is provided to "file_path".
    "query" should be specified in order to create summary,
    it is an objective of reading the pdf
    '''
    loader = PyMuPDFLoader(Path(file_path))
    docs = loader.load()
    data = "".join([doc.page_content for doc in docs])
    if len(data) > 30000 * 4:
        return summarize(query, data)
    return docs


@tool
def read_any_file(file_path: str, query: str) -> str:
    '''
    fetches and returns the summarized text content \
    of a single file (a lot of formats are supported: tables, html, documents, images),
    which path is provided to "file_path".
    "query" should be specified in order to create summary,
    it is an objective of reading the pdf
    '''
    loader = UnstructuredFileLoader(Path(file_path), mode="single")
    data = loader.load()
    if len(data[0].page_content) > 30000 * 4:
        return summarize(query, data[0].page_content)
    return data


@tool
def read_google_docs(url: str) -> str:
    '''
    use this tool if you need to read contents of the docs.google.com link

    it downloads the content of the google doc to temporary location, \
    and you'll need to read contents of the downloaded file using read_any_file tool
    '''
    splitted_url = url[url.find('docs.google.com'):].split('/')
    doc_type = splitted_url[1]
    doc_id = splitted_url[3]

    doc_type_to_format = {'document': 'docx',
                          'spreadsheets': 'xlsx',
                          'presentation': 'pptx'}

    format_ = doc_type_to_format.get(doc_type)

    if not format_:
        return "Couldn't process google doc -- invalid URL."

    download_url = f'https://docs.google.com/{doc_type}/d/{doc_id}/export?format={format_}'
    file_path = Path(f"tmp.{format_}")
    response = requests.get(download_url, timeout=10)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        return f"File downloaded successfully to {file_path.absolute()}. "\
            "Now read it using read_any_file tool."
    return "Failed to download the file."
