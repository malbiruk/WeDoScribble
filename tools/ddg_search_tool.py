import asyncio
import json
import logging

import aiohttp
import streamlit as st
from bs4 import BeautifulSoup
from charset_normalizer import detect
from duckduckgo_search import DDGS
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

llm = (ChatOpenAI(temperature=0,
                  streaming=True,
                  model_name=st.session_state['openai_model'],
                  api_key=st.secrets['OPENAI_API_KEY'])
       if st.session_state['openai_model'].startswith('gpt')
       else ChatOpenAI(temperature=0,
                       streaming=True,
                       model_name='gpt-3.5-turbo',
                       base_url='http://192.168.108.17:8001/v1',
                       api_key='sk-no-key-required'
                       ))


async def get_website_text(url: str, timeout: int = 10) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
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
        logger.error(f"failed to fetch the website 😞\n{url}\n%s", e)
        return None
    except UnicodeDecodeError as e:
        logger.error(f"failed to decode the website 😞\n{url}\n%s", e)
        return None


async def fetch_links(links):
    tasks = [get_website_text(link) for link in links]
    return await asyncio.gather(*tasks)


def ddg_search(query: str, n_results: int) -> list[dict]:
    """
    fetches and returns the full text content \
    of top n_results from DuckDuckGo for a given query
    """
    results = DDGS().text(query, max_results=n_results)
    links = [result["href"] for result in results]
    results = asyncio.run(fetch_links(links))
    return [{'contents': result, 'link': link}
            for result, link in zip(results, links)]


default_prompt = """
Write a summary of the following text for "{query}":
"{text}"
SUMMARY:
"""


def summarize(query, content, prompt=default_prompt) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=30000, chunk_overlap=100)
    docs = text_splitter.create_documents([content])
    map_prompt_template = PromptTemplate(
        template=prompt, input_variables=["text", "query"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    output = summary_chain.run(input_documents=docs, query=query)

    return output


@tool
def web_search(query: str) -> str:
    """
    fetches and returns the summarized text content \
    of top 10 results from DuckDuckGo for a given query
    along with the source URLs as json
    """
    full_pages = ddg_search(query, 10)
    return json.dumps([{'contents': summarize(query, res['contents']), 'link': res['link']}
                       for res in full_pages if res['contents']], indent=2)
