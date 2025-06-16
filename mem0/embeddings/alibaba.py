import json
import os
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class AlibabaDashScopeEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        # Set default model and dimensions for Alibaba text-embedding-v4
        self.config.model = self.config.model or "text-embedding-v4"
        self.config.embedding_dims = self.config.embedding_dims or 1024

        # Import DashScope here to avoid dependency issues if not installed
        try:
            import dashscope
            from dashscope import TextEmbedding
            self.dashscope = dashscope
            self.TextEmbedding = TextEmbedding
        except ImportError:
            raise ImportError(
                "DashScope is required for Alibaba Cloud Embedding. "
                "Install it with: pip install dashscope"
            )

        # Set API key
        api_key = self.config.api_key or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set it via config.api_key or ALIYUN_API_KEY environment variable."
            )
        
        self.dashscope.api_key = api_key

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Alibaba DashScope.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        # Clean the text
        text = text.replace("\n", " ").strip()
        
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Call DashScope TextEmbedding API
            response = self.TextEmbedding.call(
                model=self.config.model,
                input=text
            )
            
            # Check if response is successful
            if response.status_code != 200:
                raise ValueError(f"API request failed with status {response.status_code}: {response.get('message', 'Unknown error')}")
            
            # Extract embeddings from response
            if 'output' in response and 'embeddings' in response['output']:
                embeddings_data = response['output']['embeddings']
                
                # Extract the actual embedding vectors
                if isinstance(embeddings_data, list) and len(embeddings_data) > 0:
                    # Each item in embeddings_data should have an 'embedding' key
                    first_item = embeddings_data[0]
                    if isinstance(first_item, dict) and 'embedding' in first_item:
                        embedding = first_item['embedding']
                        
                        # Ensure the embedding has the correct dimensions
                        if len(embedding) != self.config.embedding_dims:
                            raise ValueError(
                                f"Expected embedding dimension {self.config.embedding_dims}, "
                                f"but got {len(embedding)}"
                            )
                        
                        return embedding
                    else:
                        return first_item
                else:
                    raise ValueError("Empty embeddings list in response")
            else:
                # Debug information
                response_keys = list(response.keys()) if hasattr(response, 'keys') else []
                output_keys = list(response.get('output', {}).keys()) if 'output' in response else []
                raise ValueError(f"No embeddings found in response. Response keys: {response_keys}, Output keys: {output_keys}")
                
        except Exception as e:
            raise Exception(f"Error calling DashScope Embedding API: {str(e)}")