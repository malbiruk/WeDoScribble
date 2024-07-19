import asyncio
import json
import logging

import aiohttp
from bs4 import BeautifulSoup
from charset_normalizer import detect
from duckduckgo_search import DDGS
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# pylint: disable = unsupported-binary-operation

logger = logging.getLogger(__name__)

llm = ChatOpenAI(temperature=0,
                 model_name='gpt-4o-mini')


async def get_website_text(url: str, timeout: int = 10) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=headers,
            fallback_charset_resolver=lambda r, b: detect(b)["encoding"] or "utf-8"
        ) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except aiohttp.ClientError as e:
        logger.warning(f"failed to fetch the website 😞\n{url}\n%s", e)
        return None
    except UnicodeDecodeError as e:
        logger.warning(f"failed to decode the website 😞\n{url}\n%s", e)
        return None
    except asyncio.TimeoutError as e:
        logger.warning(f"the website exceeded timeout 😞\n{url}\n%s", e)
        return None


async def ddg_search(query: str, n_results: int) -> list[dict]:
    """
    fetches and returns the full text content \
    of top n_results from DuckDuckGo for a given query
    """
    results = DDGS().text(query, max_results=n_results)
    links = [result["href"] for result in results]
    tasks = [get_website_text(link) for link in links]
    results = await asyncio.gather(*tasks)
    return [{'contents': result, 'link': link}
            for result, link in zip(results, links) if result]


async def extract_relevant_information(query, content) -> str:
    extract_prompt = """
Extract chunks of relevant information for "{query}" from the text below:
"{text}"
RELEVANT INFORMATION:
"""
    prompt = PromptTemplate.from_template(extract_prompt)
    extraction_chain = prompt | llm | StrOutputParser()
    if len(content) < 85000 * 4:
        return await extraction_chain.ainvoke({"text": content, "query": query})

    # don't load huge websites
    logger.warning("Encountered a huge website, won't process")
    return None


async def search_and_extract(query: str):
    full_pages = await ddg_search(query, 5)

    async def process_page_extract(res, query):
        if res['contents']:
            extracted_info = await extract_relevant_information(query, res['contents'])
            return {'contents': extracted_info, 'link': res['link']}
        return None

    tasks = [process_page_extract(res, query) for res in full_pages]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]


@tool
def web_search(query: str) -> str:
    """
    fetches and returns the relevant text content \
    from the top 5 webpages of the first web search page \
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
    fetches the content of a single webpage
    """
    return asyncio.run(get_website_text(url))
