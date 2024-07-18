import re

from langchain.tools import tool
import streamlit as st


def parse_conversation(conversation):
    messages = []
    pattern = r'(\w+):\s(.*?)\n(?=\w+:|$)'
    matches = re.findall(pattern, conversation, re.DOTALL)

    for role, content in matches:
        messages.append({'role': role, 'content': content.strip()})

    return messages


def create_export_dialogue_tool(history):
    @tool
    def export_dialogue(out_path: str) -> None:
        '''
        save messages history to out_path.

        Is used when user asks to export/save current dialogue.

        If user doesn't specify out_path, generate useful name (as a title to the current dialogue),
        but keep it lowercase and with "_" instead of spaces
        '''
        messages_history = '\n\n'.join([f'{msg.type}: {msg.content}'
                                        for msg in history.messages])

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(messages_history)

    return export_dialogue


def create_import_dialogue_tool(history):
    @tool
    def import_dialogue(file_path: str) -> str:
        '''
        import messages history from file_path

        Is used when user asks to import/load dialogue/conversation.
        '''
        try:
            with open(file_path, encoding='utf-8') as f:
                conversation = f.read()
            messages = parse_conversation(conversation)

            history.clear()
            for msg in messages:
                if msg['role'] == 'human':
                    history.add_user_message(msg['content'])
                else:
                    history.add_ai_message(msg['content'])

            st.rerun()

        except FileNotFoundError:
            return "Provided file_path does not exist!"

    return import_dialogue
