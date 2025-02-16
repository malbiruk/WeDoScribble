import re
from pathlib import Path

import streamlit as st
from langchain.tools import tool


def parse_conversation(conversation):
    lines = conversation.strip().split("\n")
    messages = []
    role = None
    content = ""
    for line in lines:
        line = line.strip()
        if line.startswith("human:"):
            if role is not None:
                messages.append({"role": role, "content": content.strip()})
            role = "human"
            content = line[6:].strip()
        elif line.startswith("ai:"):
            if role is not None:
                messages.append({"role": role, "content": content.strip()})
            role = "ai"
            content = line[3:].strip()
        elif line:
            content += "\n" + line
    if role is not None:
        messages.append({"role": role, "content": content.strip()})
    return messages


def create_export_dialogue_tool(history):
    @tool
    def export_dialogue(out_path: str) -> None:
        """
        save messages history to out_path.

        Is used when user asks to export/save current dialogue.

        If user doesn't specify out_path, generate a useful name \
        (as a title to the current dialogue),
        but keep it lowercase and with "_" instead of spaces
        """
        messages_history = "\n\n".join([f"{msg.type}: {msg.content}"
                                        for msg in history.messages])

        with Path(out_path).open("w") as f:
            f.write(messages_history)

    return export_dialogue


def create_import_dialogue_tool(history):
    @tool
    def import_dialogue(file_path: str) -> str:
        """
        import messages history from file_path

        override_current argument clears current dialogue before loading

        Is used when user asks to import/load dialogue/conversation.
        """
        try:
            with Path(file_path).open() as f:
                conversation = f.read()
            messages = parse_conversation(conversation)

            history.clear()
            for msg in messages:
                if msg["role"] == "human":
                    history.add_user_message(msg["content"])
                else:
                    history.add_ai_message(msg["content"])

            st.rerun()

        except FileNotFoundError:
            return "Provided file_path does not exist!"

        else:
            return "The conversation is successfully loaded."

    return import_dialogue
