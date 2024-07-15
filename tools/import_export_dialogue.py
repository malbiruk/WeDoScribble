from langchain.tools import tool

@tool
def export_dialogue(messages_history: str, out_path: str) -> None:
    '''
    save messages history to out_path.

    Is used when user asks to export/save current dialogue.
    You sould provide all user messages and your messages to messages_history

    If user doesn't specify out_path, generate useful name (as a title to the current dialogue),
    but keep it lowercase and with "_" instead of spaces
    '''
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(messages_history)

@tool
def import_dialogue(file_path: str) -> str:
    '''
    import messages history from file_path

    Is used when user asks to import/load dialogue/conversation.
    You should use output of this function as messages history, but don't use any other tools:
    just load user inputs and assistant outputs.
    '''
    try:
        with open(file_path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Provided file_path does not exist!"
