# WeDoScribble

WeDoScribble is an AI assistant designed to help with a variety of tasks, including web searches, file handling, weather information, Python execution, and conversation management.

## Features

- **Web Searches**: Search the web for information and get summarized results with relevant links.
- **File Handling**: Read, summarize, and write various types of files, including PDFs, documents (also publicly available google docs), tables, and images (only OCR).
- **Weather Information**: Fetch current weather information for any specified location.
- **Python Execution**: Run Python code to perform calculations, data analysis, and more.
- **Conversation Management**: Save and load conversations, allowing you to continue from where you left off.

## Name Explanation

The name "WeDoScribble" can be read as "we do scribble," emphasizing the collaborative nature of the AI assistant working together with you. Additionally, "We" stands for web, and "Do" stands for documents, highlighting the primary functionalities of the assistant.

## Repository Structure

- `app.py`: The main application script.
- `requirements.txt`: List of dependencies required to run the application.
- `logo_blue.png`: Logo image for the application.
- `tools/`: Directory containing various tools used by the application.
- `prompts/`: Directory containing prompt templates for different tasks.
- `.streamlit/`: Directory containing configuration files for Streamlit.

## Getting Started

To get started with WeDoScribble, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/wedoscribble.git
    cd wedoscribble
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install system dependencies**:
    Install the following system dependencies if they are not already available on your system with e.g. `brew install` for Mac or `apt install` for Ubuntu. Depending on what document types you're parsing, you may not need all of these.
    - `libmagic-dev` (filetype detection)
    - `poppler-utils` (images and PDFs)
    - `tesseract-ocr` (images and PDFs)
    - `qpdf` (PDFs)
    - `libreoffice` (MS Office docs)
    - `pandoc` (EPUBs)

    **Note**: two methods associated with web search and reading webpages are implemented. By default, the agent uses DuckDuckGo without browser, which is faster. However without JavaScript not all pages are available. Another option uses Google Search via Selenium, which operates headless browser instances, in which case you should also install `chromium` (`/usr/bin/chromium`).

4. **Set up API keys**:
    Create a `~/.streamlit/secrets.toml` file and add your OpenAI and OpenWeatherMap API keys:
    ```toml
    OPENAI_API_KEY = "your_openai_api_key"
    OPENWEATHERMAP_API_KEY = "your_openweathermap_api_key"
    ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Usage

Once the application is running, you can interact with WeDoScribble through the Streamlit interface. You can ask it to perform various tasks such as web searches, file handling, fetching weather information, running Python code, and managing conversations.

## Contributing

Contributions are welcome! If you have any ideas for new features or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [2601074@gmail.com](mailto:2601074@gmail.com).
