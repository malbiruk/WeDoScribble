import asyncio
import json
import logging

import chromedriver_autoinstaller
from duckduckgo_search import DDGS
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from rich.logging import RichHandler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chromedriver_autoinstaller.install()


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()],
)

logger = logging.getLogger(__name__)


llm = ChatOllama(temperature=0,
                 model="qwen2.5:7b")

default_prompt = """
Write a summary of the following text for "{query}":
"{text}"
SUMMARY:
"""

map_prompt_template = PromptTemplate(
    template=default_prompt, input_variables=["text", "query"])

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=map_prompt_template,
    verbose=False,
)

extract_prompt = """
Extract chunks of relevant information for "{query}" from the text below:
"{text}"
RELEVANT INFORMATION:
"""
prompt = PromptTemplate.from_template(extract_prompt)
extraction_chain = prompt | llm | StrOutputParser()


async def summarize(query, content) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=30000 * 4, chunk_overlap=4000)
    docs = text_splitter.create_documents([content])

    return await summary_chain.ainvoke({"input_documents": docs, "query": query})


def get_website_text(url: str) -> str:
    logger.info("started reading %s", url)
    chrome_options = Options()
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.set_script_timeout(5)
    text_content = driver.find_element(By.XPATH, "/html/body").text
    driver.close()
    logger.info("finished reading %s", url)
    return text_content


async def ddg_search(query: str) -> list[dict]:
    """
    fetches and returns the full text content \
    of top n_results from DuckDuckGo for a given query
    """
    n_results = 3
    results = DDGS().text(query, max_results=n_results)
    links = [result["href"] for result in results]
    tasks = [asyncio.to_thread(get_website_text, link) for link in links]
    results = await asyncio.gather(*tasks)
    return [{"contents": result, "link": link}
            for result, link in zip(results, links) if result]


async def extract_relevant_information(query, content) -> str:
    if len(content) < 85000 * 4:
        return await extraction_chain.ainvoke({"text": content, "query": query})

    # don't load huge websites
    logger.warning("Encountered a huge website, won't process")
    return None


async def search_and_extract(query: str):
    full_pages = await ddg_search(query)

    async def process_page_extract(res, _query):
        if res["contents"]:
            # logging.info("summarizing %s", res["link"])
            # extracted_info = await extract_relevant_information(query, res["contents"])
            # logging.info("finished summarizing %s", res["link"])
            extracted_info = res["contents"]
            return {"contents": extracted_info, "link": res["link"]}
        return None

    tasks = [process_page_extract(res, query) for res in full_pages]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]


@tool
def web_search(query: str) -> str:
    """
    fetches and returns the relevant text content \
    from the top 3 webpages of the first web search page \
    for a given query along with the source URLs as json
    """
    results = asyncio.run(search_and_extract(query))
    output = json.dumps(results, indent=2)

    while len(output) > 85000 * 4:  # drop last sites if exceeding context window
        results = results[:-1]
        output = json.dumps(results, indent=2)
        logger.warning("The context window is close to be exceeded: "
                       "dropping info from the last website")
    return output


@tool
def read_webpage(url: str) -> str:
    """
    fetches the content of a single webpage. Pass ONLY ONE URL
    """
    content = asyncio.run(get_website_text(url))
    if len(content) < 85000 * 4:
        return content
    return "Couldn't load the website contents. It's too big."
