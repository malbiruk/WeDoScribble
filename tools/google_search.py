import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def create_browser(headless: bool = True, detached: bool = False) -> webdriver:
    '''
    this function creates, starts and returns chrome browser
    (by default in headless mode)
    '''
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('headless')
    # chrome_options.add_argument('start-maximized')
    if detached:
        chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(
        service=Service(), options=chrome_options)
    return driver


def fetch_page_content(link_idx: int, search_url: str) -> dict:
    '''
    fetch the content of a single page given its link index and search URL
    '''
    driver = create_browser()
    driver.get(search_url)

    links = driver.find_elements(By.XPATH, '//cite')
    link = [link for link in links if link.text][link_idx]
    link.click()

    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '/html/body')))
        result = {'contents': driver.find_element(By.XPATH, "/html/body").text,
                  'link': driver.current_url}
        driver.quit()
        return result
    except TimeoutException:
        pass


def search_google(query: str) -> list[dict]:
    '''
    uses headless browser to obtain all text content
    from the first Google page links
    '''
    driver = create_browser()
    driver.get('https://www.google.com/')
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '//textarea')))
    search_bar = driver.find_element(By.XPATH, '//textarea')
    search_bar.send_keys(query)
    search_bar.send_keys(Keys.ENTER)
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '//cite')))
    search_url = driver.current_url
    links = driver.find_elements(By.XPATH, '//cite')
    n_true_links = len([link for link in links if link.text])
    n_links = 5 if 5 <= n_true_links else n_true_links
    driver.quit()

    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_link = {executor.submit(
            fetch_page_content, link_idx, search_url): link_idx
            for link_idx in range(n_links)}
        for future in as_completed(future_to_link):
            result = future.result()
            if result['contents']:  # Only append non-empty results
                results.append(result)

    return results


default_prompt = """
Write a summary of the following text for "{query}":
"{text}"
SUMMARY:
"""

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


def summarize(query, content, prompt=default_prompt) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=12000, chunk_overlap=1000)
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

    return summary_chain.run(input_documents=docs, query=query)


@tool
def web_search(query: str) -> str:
    """
    fetches and returns the summarized text content \
    from webpages of the first Google search page for a given query
    along with the source URLs as json
    """
    full_pages = search_google(query)
    if len('\n'.join([i['contents'] for i in full_pages])) > 14000 * 4:
        return json.dumps([{'contents': summarize(query, res['contents']), 'link': res['link']}
                           for res in full_pages if res['contents']], indent=2)

    return json.dumps(full_pages, indent=2)


@tool
def read_webpage(url: str, query: str) -> str:
    """
    fetches the content of a single webpage, if the page is very long,
    this function returns summarized result using "query"
    """
    driver = create_browser()
    driver.get(url)
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '/html/body')))
    content = driver.find_element(By.XPATH, "/html/body").text
    if len(content) > 14000 * 4:
        return summarize(query, content)
    return content
