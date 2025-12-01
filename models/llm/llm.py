import logging
from typing import Optional, Generator, Union
from collections.abc import Mapping
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
import requests

logger = logging.getLogger(__name__)


class CerebrasLargeLanguageModel(LargeLanguageModel):
    """
    Cerebras large language model implementation
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model
        
        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        api_key = credentials.get("cerebras_api_key")
        api_base = credentials.get("cerebras_api_base", "https://api.cerebras.ai/v1")
        
        if not api_key:
            raise CredentialsValidateFailedError("Cerebras API key is required")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Convert prompt messages to Cerebras format
        messages = []
        for message in prompt_messages:
            messages.append({
                "role": message.role.value,
                "content": message.content
            })
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **model_parameters
        }
        
        if stop:
            payload["stop"] = stop
        
        # Make API request
        url = f"{api_base}/chat/completions"
        
        try:
            if stream:
                return self._handle_stream_response(url, headers, payload)
            else:
                return self._handle_sync_response(url, headers, payload)
        except Exception as e:
            logger.exception(f"Error invoking Cerebras model: {e}")
            raise InvokeError(str(e))
    
    def _handle_sync_response(self, url: str, headers: dict, payload: dict) -> LLMResult:
        """Handle synchronous response"""
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        return LLMResult(
            model=result.get("model", payload["model"]),
            prompt_messages=[],
            message=result["choices"][0]["message"],
            usage={
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"]
            }
        )
    
    def _handle_stream_response(self, url: str, headers: dict, payload: dict) -> Generator:
        """Handle streaming response"""
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if not line_str.startswith('data: '):
                continue
            
            data_str = line_str[6:]  # Remove 'data: ' prefix
            if data_str == '[DONE]':
                break
            
            try:
                import json
                chunk_data = json.loads(data_str)
                
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    choice = chunk_data["choices"][0]
                    delta = choice.get("delta", {})
                    
                    if "content" in delta:
                        yield LLMResultChunk(
                            model=chunk_data.get("model", payload["model"]),
                            prompt_messages=[],
                            delta=LLMResultChunkDelta(
                                index=choice.get("index", 0),
                                message=delta
                            )
                        )
            except json.JSONDecodeError:
                continue
    
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages
        
        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return: number of tokens
        """
        # Simple token estimation (4 chars ≈ 1 token)
        total_text = ""
        for message in prompt_messages:
            total_text += message.content
        
        return len(total_text) // 4
    
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials
        
        :param model: model name
        :param credentials: model credentials
        :raises CredentialsValidateFailedError: if validation failed
        """
        try:
            # Make a simple test request
            api_key = credentials.get("cerebras_api_key")
            api_base = credentials.get("cerebras_api_base", "https://api.cerebras.ai/v1")
            
            if not api_key:
                raise CredentialsValidateFailedError("Cerebras API key is required")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": False
            }
            
            url = f"{api_base}/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status {response.status_code}: {response.text}"
                )
        except requests.RequestException as e:
            raise CredentialsValidateFailedError(f"Credentials validation failed: {str(e)}")
