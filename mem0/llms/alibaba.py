import json
import os
from typing import Dict, List, Optional

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class AlibabaDashScopeLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "qwen-plus"

        # Import DashScope here to avoid dependency issues if not installed
        try:
            import dashscope
            from dashscope import Generation
            self.dashscope = dashscope
            self.Generation = Generation
        except ImportError:
            raise ImportError(
                "DashScope is required for Alibaba Cloud LLM. "
                "Install it with: pip install dashscope"
            )

        # Set API key
        api_key = self.config.api_key or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set it via config.api_key or ALIYUN_API_KEY environment variable."
            )
        
        self.dashscope.api_key = api_key

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from DashScope API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            # For tool calls, we need to handle the response differently
            # DashScope may have different response format for tool calls
            processed_response = {
                "content": response.output.text if hasattr(response.output, 'text') else str(response.output),
                "tool_calls": [],
            }
            
            # Check if there are tool calls in the response
            if hasattr(response.output, 'tool_calls') and response.output.tool_calls:
                for tool_call in response.output.tool_calls:
                    processed_response["tool_calls"].append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    })
            
            return processed_response
        else:
            # For regular text responses
            if hasattr(response.output, 'text'):
                return response.output.text
            else:
                return str(response.output)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Alibaba DashScope.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        # Convert messages to DashScope format
        dashscope_messages = []
        for message in messages:
            dashscope_messages.append({
                "role": message["role"],
                "content": message["content"]
            })

        # Prepare parameters for DashScope API
        params = {
            "model": self.config.model,
            "messages": dashscope_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            if tool_choice != "auto":
                params["tool_choice"] = tool_choice

        # Make the API call
        try:
            response = self.Generation.call(**params)
            
            # Check if the response is successful
            if response.status_code == 200:
                return self._parse_response(response, tools)
            else:
                raise Exception(f"DashScope API error: {response.message}")
                
        except Exception as e:
            raise Exception(f"Error calling DashScope API: {str(e)}")