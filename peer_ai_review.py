import os
import logging
import re
import abc
from typing import Optional, List, Dict, Any, NamedTuple, Type
from difflib import get_close_matches
import logging
import sys
from typing_extensions import Annotated
import argparse

from fastmcp import FastMCP

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI, OpenAIError = None, Exception
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    genai, google_exceptions = None, None
try:
    import requests
except ImportError:
    requests = None


class CompletionResponse(NamedTuple):
    content: str

class LLMProvider(abc.ABC):
    def __init__(self, fallback_model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        if not fallback_model:
            raise ValueError("A fallback_model must be provided.")
        self.fallback_model = fallback_model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def get_completion(self, user_question: str, system_prompt: str, model: str, max_retries: int = 3, **params) -> CompletionResponse:
        pass

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        pass
    
    def find_best_model_match(self, user_input: str) -> str:
        try:
            model_ids = self.list_models()
            if not model_ids:
                self.logger.warning("Model list is empty. Using fallback.")
                return self.fallback_model
            
            matches = get_close_matches(user_input, model_ids, n=1, cutoff=0.4)
            if matches:
                self.logger.info(f"Fuzzy match for '{user_input}': '{matches[0]}'")
                return matches[0]
            else:
                self.logger.warning(f"No match for '{user_input}'. Falling back.")
                return self.fallback_model
        except Exception as e:
            self.logger.error(f"Failed to match models, using fallback: {e}")
            return self.fallback_model


class OpenAIProvider(LLMProvider):
    def __init__(self, fallback_model: str = "gpt-4o-mini", **kwargs):
        super().__init__(fallback_model=fallback_model, **kwargs)
        if OpenAI is None:
            raise ImportError("OpenAI SDK not installed. Please install with `pip install .[openai]`")
        
        resolved_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=resolved_key, base_url=self.base_url, timeout=self.timeout)

    def get_completion(self, user_question, system_prompt, model, max_retries=3, **params) -> CompletionResponse:
        full_prompt = f"{system_prompt}\n\n[USER QUESTION]\n{user_question}"
        attempt = 0
        
        while True:
            try:
                response = self.client.responses.create(
                    model=model, 
                    input=full_prompt, 
                    **params
                )
                content = "".join(chunk.text for block in response.output for chunk in block.content)
                return CompletionResponse(content.strip())
            except OpenAIError as e:
                attempt += 1
                self.logger.warning(f"OpenAI error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    raise

    def list_models(self) -> List[str]:
        self.logger.debug("Fetching available models from OpenAI")
        models = [model.id for model in self.client.models.list().data]
        self.logger.debug(f"Found {len(models)} OpenAI models: {', '.join(models)}")
        return models


class AnthropicProvider(LLMProvider):
    def __init__(self, fallback_model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(fallback_model=fallback_model, **kwargs)
        if anthropic is None:
            raise ImportError("Anthropic SDK not installed. Please install with `pip install .[anthropic]`")
        
        resolved_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("Missing Anthropic API key. Set the ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=resolved_key, timeout=self.timeout)

    def get_completion(self, user_question, system_prompt, model, max_retries=3, **params) -> CompletionResponse:
        messages = [{"role": "user", "content": user_question}]
        attempt = 0
        
        while True:
            try:
                response = self.client.messages.create(
                    model=model, 
                    system=system_prompt, 
                    messages=messages,
                    **params
                )
                content = response.content[0].text
                return CompletionResponse(content.strip())
            except anthropic.APIError as e:
                attempt += 1
                self.logger.warning(f"Anthropic error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    raise

    def list_models(self) -> List[str]:
        self.logger.debug("Fetching available models from Anthropic")
        try:
            model_page = self.client.models.list()
            model_ids = [model.id for model in model_page.data]
            self.logger.debug(f"Found {len(model_ids)} Anthropic models: {', '.join(model_ids)}")
            return model_ids
        except anthropic.APIError as e:
            self.logger.error(f"Failed to fetch models from Anthropic SDK: {e}")
            return []


class GeminiProvider(LLMProvider):
    def __init__(self, fallback_model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(fallback_model=fallback_model, **kwargs)
        if genai is None:
            raise ImportError("Google SDK not installed. Please install with `pip install .[gemini]`")
        
        resolved_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing Google API key. Set the GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=resolved_key)

    def get_completion(self, user_question, system_prompt, model, max_retries=3, **params) -> CompletionResponse:
        client = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
        generation_config = genai.types.GenerationConfig(**params)
        attempt = 0
        
        while True:
            try:
                response = client.generate_content(user_question, generation_config=generation_config)
                return CompletionResponse(response.text.strip())
            except (google_exceptions.GoogleAPICallError, google_exceptions.RetryError) as e:
                attempt += 1
                self.logger.warning(f"Google API error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    raise

    def list_models(self) -> List[str]:
        self.logger.debug("Fetching available models from Google Gemini")
        try:
            models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            self.logger.debug(f"Found {len(models)} Gemini models: {', '.join(models)}")
            return models
        except Exception as e:
            self.logger.error(f"Failed to fetch model list from Google: {e}")
            return []


class DeepSeekProvider(LLMProvider):
    def __init__(self, fallback_model: str = "deepseek-chat", **kwargs):
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        super().__init__(fallback_model=fallback_model, **kwargs)
        if OpenAI is None:
            raise ImportError("OpenAI SDK not installed (required for DeepSeek). Please install with `pip install .[openai]`")
        
        resolved_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError("Missing DeepSeek API key. Set the DEEPSEEK_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=resolved_key, base_url=self.base_url, timeout=self.timeout)

    def get_completion(self, user_question, system_prompt, model, max_retries=3, **params) -> CompletionResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        attempt = 0
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=params.get("temperature", 0.7),
                    **{k: v for k, v in params.items() if k not in ["max_tokens", "temperature"]}
                )
                content = response.choices[0].message.content
                return CompletionResponse(content.strip())
            except OpenAIError as e:
                attempt += 1
                self.logger.warning(f"DeepSeek error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    raise

    def list_models(self) -> List[str]:
        self.logger.debug("Fetching available models from DeepSeek")
        try:
            models = [model.id for model in self.client.models.list().data]
            self.logger.debug(f"Found {len(models)} DeepSeek models: {', '.join(models)}")
            return models
        except OpenAIError as e:
            self.logger.error(f"Failed to fetch models from DeepSeek: {e}")
            fallback_models = ["deepseek-chat", "deepseek-coder"]
            self.logger.info(f"Using fallback model list: {fallback_models}")
            return fallback_models


class OllamaProvider(LLMProvider):
    def __init__(self, fallback_model: str = "llama3", ollama_host: str = None, ollama_port: str = None, **kwargs):
        host = ollama_host or os.getenv("OLLAMA_HOST", "localhost")
        port = ollama_port or os.getenv("OLLAMA_PORT", "11434")
        default_base_url = f"http://{host}:{port}"
        kwargs.setdefault("base_url", default_base_url)
        super().__init__(fallback_model=fallback_model, **kwargs)
        if requests is None:
            raise ImportError("Requests library not installed. Please install with `pip install .[ollama]`")
        
        self.logger.info(f"Connecting to Ollama at {self.base_url}")
        try:
            self.list_models()
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Error: {e}")

    def get_completion(self, user_question, system_prompt, model, max_retries=3, **params) -> CompletionResponse:
        ollama_options = params.copy()
        payload = {
            "model": model,
            "prompt": user_question,
            "system": system_prompt,
            "stream": False,
            "options": ollama_options
        }
        url = f"{self.base_url}/api/generate"
        attempt = 0
        
        while True:
            try:
                resp = requests.post(url, json=payload, timeout=(5, 60))
                resp.raise_for_status()
                json_response = resp.json()
                content = json_response.get("response", "")
                return CompletionResponse(content.strip())
            except requests.exceptions.RequestException as e:
                attempt += 1
                self.logger.warning(f"Ollama request error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    raise

    def list_models(self) -> List[str]:
        url = f"{self.base_url}/api/tags"
        self.logger.info(f"Fetching available models from Ollama at {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models if "name" in m]
            self.logger.info(f"Found {len(model_names)} Ollama models: {', '.join(model_names)}")
            return model_names
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch models from Ollama at %s: %s", url, e)
            return []


class ProviderFactory:
    def __init__(self, cli_args=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._providers: Dict[str, Any] = {}
        self.cli_args = cli_args or {}

    def register(self, name: str, provider_class: Type[LLMProvider], aliases: List[str], fallback: str):
        self._providers[name] = {"class": provider_class, "aliases": aliases, "fallback": fallback}

    def get_provider(self, name_query: str, **kwargs: Any) -> LLMProvider:
        query_lower = name_query.lower()
        for name, data in self._providers.items():
            if any(alias in query_lower for alias in data["aliases"]):
                self.logger.info(f"Matched query '{name_query}' to provider '{name}'.")
                ProviderClass = data["class"]
                config = {"fallback_model": data["fallback"], **kwargs}
                
                if name == "openai" and self.cli_args.get("openai_key"):
                    config["api_key"] = self.cli_args["openai_key"]
                    self.logger.debug("got openai API key")
                elif name == "anthropic" and self.cli_args.get("claude_key"):
                    config["api_key"] = self.cli_args["claude_key"]
                    self.logger.debug("got anthropic API key")
                elif name == "gemini" and self.cli_args.get("gemini_key"):
                    config["api_key"] = self.cli_args["gemini_key"]
                    self.logger.debug("got gemini API key")
                elif name == "deepseek" and self.cli_args.get("deepseek_key"):
                    config["api_key"] = self.cli_args["deepseek_key"]
                    self.logger.debug("got deepseek API key")
                elif name == "ollama":
                    if self.cli_args.get("ollama_host"):
                        config["ollama_host"] = self.cli_args["ollama_host"]
                    if self.cli_args.get("ollama_port"):
                        config["ollama_port"] = self.cli_args["ollama_port"]

                return ProviderClass(**config)
        raise ValueError(f"Unknown provider query: '{name_query}'. Known: {list(self._providers.keys())}")
    

global_cli_args = {}

mcp = FastMCP("peer-AI-review",
              instructions="""
                  This tool sends your question and answer to a peer AI for review.
                  the first parameter is the AI provider, such as openai, google gemini, anthropic claude, deepseek, ollama,
                  and set a dedicated model of the provider as the second input,
                  the third parameter is user's question and current AI's response,
                  this tool will generate a review result for your reference.
              """)

def setup_logging(level: int = logging.DEBUG):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logging.getLogger("fastmcp").setLevel(logging.DEBUG)


@mcp.tool()
def peer_ai_review(
    provider_query: Annotated[str, "Keyword for the provider (e.g., openai, claude, gemini, deepseek, ollama)."],
    model_query: Annotated[str, "Keyword for the model (e.g., 'gpt-4o-mini', 'sonnet', 'llama3')."],
    question_and_answer: Annotated[str, "The summary of the question and answer to be reviewed."]
    ):
    """
    Sends your question and answer to a peer AI for review and gets the result.
    the first parameter is AI provider, such as openai, gemini, ollama, claude, deepseek,
    the second parameter is model for the provider, such as 4o-mini for openai, qwen for ollama,
    the third parameter is the user question and current AI's answer    
    """
    logger = logging.getLogger(__name__)
    
    PROMPT_PREFIX = "You are an expert peer reviewer. Analyze the following question and answer, then verify the correctness of answer and provide a concise, critical review."

    # **MODIFIED: Pass CLI arguments to factory**
    factory = ProviderFactory(global_cli_args)
    factory.register("openai", OpenAIProvider, ["openai", "gpt", "chatgpt"], "gpt-4o-mini")
    factory.register("anthropic", AnthropicProvider, ["anthropic", "claude"], "claude-3-haiku-20240307")
    factory.register("gemini", GeminiProvider, ["gemini", "google"], "gemini-1.5-flash-latest")
    factory.register("deepseek", DeepSeekProvider, ["deepseek", "deep-seek"], "r1")
    factory.register("ollama", OllamaProvider, ["ollama", "local"], "llama3")

    try:
        provider = factory.get_provider(provider_query)
        model = provider.find_best_model_match(user_input=model_query)
        logger.info(f"Resolved to use model: {provider} {model}, send for review and wait for result")

        response = provider.get_completion(
            user_question=question_and_answer,
            system_prompt=PROMPT_PREFIX,
            model=model,
            temperature=0.7
        )

        cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()        
        logger.debug("--- AI Peer Review Result ---")
        logger.debug(f"\n{cleaned_content}")
        logger.debug("-----------------------------")
        logger.info("Peer review completed")
        return "here is the review result, read it and modify previous answer if necessary: " + cleaned_content


    except Exception as e:
        error_message = f"An error occurred during peer review: {e}, show this message to user, need user to check and modify accordingly"
        logger.error(error_message)
        return error_message


def parse_args():
    parser = argparse.ArgumentParser(description="Peer AI Review MCP Server")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--claude-key", type=str, help="Anthropic Claude API key")
    parser.add_argument("--gemini-key", type=str, help="Google API key for Gemini")
    parser.add_argument("--deepseek-key", type=str, help="DeepSeek API key")
    parser.add_argument("--ollama-host", type=str, help="Ollama server host")
    parser.add_argument("--ollama-port", type=str, help="Ollama server port")
    return parser.parse_args()


def main():
    global global_cli_args
    args = parse_args()
    global_cli_args = {
        "openai_key": args.openai_key,
        "claude_key": args.claude_key,
        "gemini_key": args.gemini_key,
        "deepseek_key": args.deepseek_key,
        "ollama_host": args.ollama_host,
        "ollama_port": args.ollama_port,
    }
    
    setup_logging()
    mcp.run()

if __name__ == "__main__":
    main()