from openai import OpenAI
from src import config

def get_llm_client() -> OpenAI:
    """
    Factory function to initialize OpenAI client based on configuration.
    Supports both OpenAI and Groq APIs dynamically.
    """
    api_key = config.LLM_API_KEY
    base_url = config.LLM_BASE_URL
    
    if not api_key:
        # Raise an informative exception if no API Key is found
        raise ValueError(
            "API Key is missing! Please configure LLM_API_KEY in your .env file."
        )
        
    if not base_url:
        if config.LLM_PROVIDER == "groq":
            base_url = "https://api.groq.com/openai/v1"
        else:
            base_url = "https://api.openai.com/v1"
            
    return OpenAI(api_key=api_key, base_url=base_url)

def get_llm_model() -> str:
    """
    Returns the configured model name.
    """
    return config.LLM_MODEL

import time
import re

def call_llm(prompt: str, temperature: float = 0.1, max_retries: int = 7) -> str:
    """
    Unified LLM caller with robust exponential backoff retry logic.
    Handles rate-limits (HTTP 429 / RESOURCE_EXHAUSTED) gracefully.
    """
    client = get_llm_client()
    model = get_llm_model()
    delay = 5.0
    
    for attempt in range(max_retries):
        try:
            # Proactive pause to avoid hitting rate limits
            time.sleep(config.LLM_DELAY)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            print(f"[!] LLM API error (attempt {attempt+1}/{max_retries}): {e}")
            
            # Check for rate limit indicators (429, quota, exhausted)
            is_rate_limit = ("429" in err_str or 
                             "rate_limit" in err_str.lower() or 
                             "quota" in err_str.lower() or 
                             "resource_exhausted" in err_str.lower())
            
            if is_rate_limit and attempt < max_retries - 1:
                # Try to parse retry delay from error message
                # Example: "Please retry in 44.865188474s."
                retry_match = re.search(r'retry in ([\d\.]+)s', err_str)
                if retry_match:
                    sleep_time = float(retry_match.group(1)) + 2.0
                else:
                    sleep_time = delay * (2 ** attempt)
                    
                print(f"[*] Quota/Rate Limit hit. Sleeping for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
            elif attempt < max_retries - 1:
                # Other errors - sleep briefly and retry
                time.sleep(3.0)
            else:
                # Raise error if retries exhausted
                raise e

