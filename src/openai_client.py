"""OpenAI Client for AlphaEvolve using OpenRouter or Gemini API."""

import os
from openai import OpenAI

class OpenAIClient:
    """Client for interacting with OpenAI models via OpenRouter or Gemini API."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gemini-2.5-flash"):
        """Initialize the OpenAI client for Gemini API."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as GEMINI_API_KEY environment variable")
        
        # Use Gemini API base URL
        base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )
        self.model = model
        
    
    def generate_completion(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Generate a completion from the model based on the provided messages."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            return ""

    async def generate(self, user_prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 4096):
        """Generate a response from the model based on user and system prompts."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate_completion(messages, temperature, max_tokens)

    async def batch_generate(self, prompts: list, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 4096):
        """Generate responses for a batch of prompts."""
        import asyncio
        tasks = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            tasks.append(asyncio.to_thread(self.generate_completion, messages, temperature, max_tokens))
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close any resources or connections if necessary."""
        pass

    def extract_code(self, content: str) -> str:
        """Extract code from the response content, assuming markdown format with code blocks."""
        import re
        code_blocks = re.findall(r'```(?:\w*)\s*([\s\S]*?)\s*```', content)
        if code_blocks:
            return code_blocks[0].strip()
        return content
