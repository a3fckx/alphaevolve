"""Client for AlphaEvolve using Gemini API for text generation."""

import os
from typing import List
from google import genai
from google.genai import types

class OpenAIClient:
    """Client for interacting with Gemini API for text generation."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gemini-2.5-flash-lite-preview-06-17"):
        """Initialize the client for Gemini API."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as GEMINI_API_KEY environment variable")
        
        # Initialize the GenAI client
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        
    def generate_completion(self, messages: List[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Generate a completion from the model based on the provided messages."""
        try:
            # Prepare the content for Gemini API
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                contents.append({"role": role if role != "system" else "model", "parts": [{"text": content_text}]})
            
            # Generate content using the GenAI client
            response = self.client.models.generate_content(
                model=f"models/{self.model}",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            
            if not response.candidates or not response.candidates[0].content:
                print(f"Warning: No valid response content received from API")
                return ""
                
            content = response.candidates[0].content.parts[0].text
            if not content:
                print(f"Warning: Empty response content received from API")
            else:
                print(f"Debug: Response content received (first 200 chars): {content[:200]}...")
            return content
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            return ""

    async def generate(self, user_prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 4096):
        """Generate a response from the model based on user and system prompts."""
        import asyncio
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(None, functools.partial(self.generate_completion, messages, temperature, max_tokens))

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
