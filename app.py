'''
WeDoScribble

web + doc + scribble
'''

import os

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from PIL import Image
from tools.import_export_dialogue import export_dialogue, import_dialogue

os.environ["OPENWEATHERMAP_API_KEY"] = st.secrets['OPENWEATHERMAP_API_KEY']


def customize_page_appearance() -> None:
    '''
    remove streamlit's red bar and 'deploy' button at the top
    '''
    st.markdown('''
        <style>
            [data-testid='stDecoration'] {
                display: none;
            }
            .stDeployButton {
                visibility: hidden;
            }
        </style>
        ''', unsafe_allow_html=True
                )


chat_avatars = {
    'user': ':material/arrow_forward_ios:',
    'human': ':material/arrow_forward_ios:',
    'assistant': ':material/psychology:',
    'ai': ':material/psychology:'
}


def init_session_state():
    if 'openai_model' not in st.session_state:
        st.session_state['openai_model'] = ''


def main():
    # page customizations
    init_session_state()
    im = Image.open("logo_blue.png")
    st.set_page_config(page_title='WeDoScribble',
                       page_icon=im)
    customize_page_appearance()
    st.markdown('<h1>WeDo<span style="color: #4285f4;">Scribble</span></h1>',
                unsafe_allow_html=True)

    # setup agent
    with open('prompts/system_prompt.txt', encoding='utf-8') as f:
        system_prompt = f.read()

    st.session_state['openai_model'] = st.selectbox('Choose a model',
                                                    ('gpt-4o', 'gpt-3.5-turbo'))

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

    # from tools.google_search import read_webpage, web_search  # Google Search via Selenium
    from tools.web_search import read_webpage, web_search  # DuckDuckGo without browser
    from tools.read_files import read_any_file, read_pdf

    tools = load_tools(["openweathermap-api"]) + [
        web_search, read_webpage, read_pdf, read_any_file, export_dialogue, import_dialogue,
        PythonREPLTool(), PubmedQueryRun()
    ] + FileManagementToolkit(
        root_dir=str('/home/klim/'),
        selected_tools=["read_file", "list_directory", "file_search"],
    ).get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    # agent = CustomAgent(llm, tools, system_prompt).agent

    history = StreamlitChatMessageHistory(key="chat_history")
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    for msg in history.messages:
        st.chat_message(msg.type, avatar=chat_avatars[msg.type]).write(msg.content)

    # handle user message
    if prompt := st.chat_input('What is up?'):
        st.chat_message("user", avatar=chat_avatars["user"]).write(prompt)
        with st.chat_message("assistant", avatar=chat_avatars["assistant"]):
            thoughts = st.container()
            st_callback = StreamlitCallbackHandler(thoughts)
            response = agent_with_chat_history.invoke(
                {"input": prompt},
                config={"callbacks": [st_callback],
                        "configurable": {"session_id": "any"}},
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
