from langchain.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from tools.google_search import summarize
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


@tool
def read_pdf(file_path: str, query: str) -> str:
    '''
    fetches and returns the summarized text content \
    of a single .pdf file, which path is provided to "file_path".
    "query" should be specified in order to create summary,
    it is an objective of reading the pdf
    '''
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    data = "".join([doc.page_content for doc in docs])
    if len(data) > 14000 * 4:
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
    loader = UnstructuredFileLoader(file_path, mode="single")
    data = loader.load()
    if len(data[0].page_content) > 14000 * 4:
        return summarize(query, data[0].page_content)
    return data
