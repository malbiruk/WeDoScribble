"""
WeDoScribble

web + doc + scribble
"""

import logging
import os
from pathlib import Path

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.tools import PythonREPLTool
from PIL import Image
from rich.logging import RichHandler
from tools.web_search import llm

logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

os.environ["OPENWEATHERMAP_API_KEY"] = st.secrets["OPENWEATHERMAP_API_KEY"]


def customize_page_appearance() -> None:
    """
    remove streamlit"s red bar and "deploy" button at the top
    """
    st.markdown("""
        <style>
            [data-testid="stDecoration"] {
                display: none;
            }
            .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True)


chat_avatars = {
    "user": ":material/arrow_forward_ios:",
    "human": ":material/arrow_forward_ios:",
    "assistant": ":material/psychology:",
    "ai": ":material/psychology:",
}


def main():
    im = Image.open("/home/klim/Documents/work/scripts/wedoscribble/logo_blue.png")
    st.set_page_config(page_title="WeDoScribble",
                       page_icon=im)
    customize_page_appearance()
    st.markdown('<h1>WeDo<span style="color: #4285f4;">Scribble</span></h1>',
                unsafe_allow_html=True)

    # setup agent
    with Path("/home/klim/Documents/work/scripts/"
              "wedoscribble/prompts/system_prompt.txt").open() as f:
        system_prompt = f.read()

    from tools.import_export_dialogue import (create_export_dialogue_tool,
                                              create_import_dialogue_tool)
    from tools.read_files import (read_any_file, read_google_docs, read_pdf,
                                  see_image)
    from tools.web_search import read_webpage, web_search

    history = StreamlitChatMessageHistory(key="chat_history")

    export_dialogue = create_export_dialogue_tool(history)
    import_dialogue = create_import_dialogue_tool(history)

    tools = [*load_tools(["openweathermap-api"]), *[
        web_search, read_webpage, read_pdf, see_image, read_any_file, read_google_docs,
        export_dialogue, import_dialogue,
        PythonREPLTool(),
    ], *FileManagementToolkit(
        root_dir="/home/klim/",
        selected_tools=["write_file", "list_directory", "file_search"],
    ).get_tools()]

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    # ])
    # agent = create_react_agent(llm, tools, prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # First put the history
            ("placeholder", "{chat_history}"),
            # Then the new input
            ("human", "{input}"),
            # Finally the scratchpad
            ("placeholder", "{agent_scratchpad}"),
        ],
    )
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda _session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    for msg in history.messages:
        st.chat_message(msg.type, avatar=chat_avatars[msg.type]).write(msg.content)

    # handle user message
    if prompt := st.chat_input("What is up?"):
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


if __name__ == "__main__":
    main()
