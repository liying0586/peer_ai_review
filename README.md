# peer_ai_review

A MCP tool to get AI-powered reviews for your questions and answers.

This tool allows you to easily send question and answer to various leading AI models (OpenAI, Anthropic, Google Gemini, and local models via Ollama) and receive peer review result.

Multi-Provider Support:

    OpenAI (openai, gpt)

    Anthropic (anthropic, claude)
    
    Google (gemini, google)
    
    Ollama (ollama, local) for running local models.

Fuzzy Model Matching:

You don't need the exact model name; the tool finds the closest match (e.g., "4o-mini" works for "gpt-4o-mini").

# Prerequisites

Python 3.12+

Install uv

# Installation

download this project and install dependencies

```bash
pip install ".[all]"
```

if you only want to install necessary dependencies, you can replace [all] with provider you want to use, check pyproject.toml for details, here is an example

```bash
pip install ".[openai]"
```

# Configuration

You can configure API keys and other settings in two ways. Command-line flags always take precedence over environment variables.

1. By Config

Example for OpenAI

```bash
--openai-key YOUR_API_KEY
```

Example for Ollama

```bash
--ollama-host 192.168.1.100 --ollama-port 11434
```

Configuration Example

here is the configuration example
```bash
"peer_ai_review": {
  "command": "uv",
  "args": [
    "run",
    "path_to/peer_ai_review.py",
    "--openai-key",
    "sk-xxx",
    "--gemini-key",
    "AIxxx",
    "--deepseek-key",
    "sk-xxx",
    "--ollama-host",
    "192.168.1.100",
    "--ollama-port",
    "11434"
  ]
}
```

2. By Environment Variables

Set environment variables in your shell or .env file. The tool will automatically detect them.

For OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

For Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

For Google

```bash
export GOOGLE_API_KEY="AIzaSy..."
```

For a custom Ollama server

```bash
export OLLAMA_HOST="192.168.1.100"
export OLLAMA_PORT="11434"
```



# Usage
After install and enable this MCP server in your AI environment (cursor/vscode, webUI), use prompt to AI to send peer view, here is an example "send above question and answer to gemini flash for review"

