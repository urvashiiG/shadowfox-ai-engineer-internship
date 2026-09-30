from openai import OpenAI
from app.config.settings import get_settings

class LLMError(RuntimeError):
    """Safe external LLM configuration or request error."""

class OpenRouterLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None, client=None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = base_url or settings.openrouter_base_url
        self.client = client

    def complete(self, prompt: str) -> str:
        if not self.api_key and self.client is None:
            raise LLMError("OpenRouter is not configured. Add OPENROUTER_API_KEY to the environment.")
        try:
            client = self.client or OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)
            response = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise LLMError("The language model returned an empty response. Please retry.")
            return answer
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("The language model could not complete the request. Check configuration or retry later.") from exc
