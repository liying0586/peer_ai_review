# peer_ai_review
A versatile, multi-provider MCP tool to get AI-powered reviews for your questions and answers. This tool allows you to easily send a summary to various leading AI models (OpenAI, Anthropic, Google Gemini, and local models via Ollama) and receive peer review.
Multi-Provider Support: Seamlessly switch between different AI providers:
OpenAI (openai, gpt)
Anthropic (anthropic, claude)
Google (gemini, google)
Ollama (ollama, local) for running local models.
Fuzzy Model Matching: You don't need the exact model name; the tool finds the closest match (e.g., "4o-mini" works for "gpt-4o-mini").
Prerequisites
Python 3.12+
Install uv
Installation
Generated bash
pip install ".[all]"
bash
if you only want to install necessary dependencies, you can replace all with provider you want to use, check pyproject.toml for details, here is an example
Generated bash
pip install ".[openai]"
bash

Configuration
You can configure API keys and other settings in two ways. Command-line flags always take precedence over environment variables.
1. By Command-Line Flags (Recommended for quick tests)
Pass configuration directly when you run the script.
Generated bash
# Example for OpenAI
--openai-key YOUR_API_KEY

# Example for Ollama
--ollama-host 192.168.1.100 --ollama-port 11434
Use code with caution.
Bash
2. By Environment Variables (Recommended for regular use)
Set environment variables in your shell or .env file. The tool will automatically detect them.
Generated bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For Google
export GOOGLE_API_KEY="AIzaSy..."

# For a custom Ollama server
export OLLAMA_HOST="192.168.1.100"
export OLLAMA_PORT="11434"
Use code with caution.
Bash
Usage
The script is run using uv run followed by the path to the script and its arguments. The basic syntax is:
Generated bash
uv run peer_ai_review.py <provider> <model> "<question_and_answer_summary>"
Bash
<provider>: A keyword for the provider (e.g., ollama, openai).
<model>: A keyword for the model (e.g., llama3, gpt-4o-mini).
<question_and_answer_summary>: The text you want the AI to review, enclosed in quotes.
Examples
Using Ollama (Local Model)
This is the simplest use case as it typically requires no API key. Make sure your Ollama server is running.
Generated bash
uv run ollama peer_ai_review.py ollama llama3 "is 9997 prime number? yes"
Bash
If your Ollama server is on a different host:
Generated bash
uv run ollama peer_ai_review.py --ollama-host 10.1.1.50 ollama llama3 "is 9997 prime number? yes"
Use code with caution.
Bash
Using OpenAI
You must install the openai extra and provide an API key.
Generated bash
# Using a command-line flag for the key
uv run openai peer_ai_review.py --openai-key sk-your-key-here openai "4o-mini" "is 9997 prime number? yes"

# Or, if you have set the OPENAI_API_KEY environment variable
uv run openai peer_ai_review.py openai gpt-4o-mini "is 9997 prime number? yes"
Bash
Using Google Gemini
You must install the gemini extra and provide an API key.
Generated bash
uv run gemini peer_ai_review.py --google-key AIzaSy... google flash "is 9997 prime number? yes"
Bash
Using Anthropic Claude
You must install the anthropic extra and provide an API key.
Generated bash
uv run peer_ai_review.py claude haiku "is 9997 prime number? yes"
Bash
