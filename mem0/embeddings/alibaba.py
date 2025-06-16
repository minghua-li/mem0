import json
import os
from typing import Literal, Optional
import logging # Added logging

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__) # Added logger

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
        logger.debug(f"AlibabaDashScopeEmbedding: Attempting to embed text: '{text[:100]}...' for action: {memory_action}") # Added log
        # Clean the text
        text = text.replace("\\n", " ").strip()

        if not text:
            logger.error("AlibabaDashScopeEmbedding: Text cannot be empty") # Added log
            raise ValueError("Text cannot be empty")

        try:
            # Call DashScope TextEmbedding API
            logger.debug(f"AlibabaDashScopeEmbedding: Calling DashScope TextEmbedding API with model: {self.config.model}, requested dimension: {self.config.embedding_dims}") # Added log
            response = self.TextEmbedding.call(
                model=self.config.model,
                input=text,
                dimension=self.config.embedding_dims # Added dimension parameter
            )
            logger.debug(f"AlibabaDashScopeEmbedding: Received response from DashScope API. Status: {response.status_code}") # Added log
            # Log the full response if it's not too large, or a summary
            try:
                response_content_str = str(response) # Or json.dumps(response) if it's a dict-like object
                if len(response_content_str) > 500:
                    logger.debug(f"AlibabaDashScopeEmbedding: DashScope API response (summary): {response_content_str[:500]}...")
                else:
                    logger.debug(f"AlibabaDashScopeEmbedding: DashScope API response: {response_content_str}")
            except Exception as e_log_resp:
                logger.warning(f"AlibabaDashScopeEmbedding: Could not serialize/log full API response: {e_log_resp}")


            # Check if response is successful
            if response.status_code != 200:
                logger.error(f"AlibabaDashScopeEmbedding: API request failed with status {response.status_code}: {response.get('message', 'Unknown error')}") # Added log
                raise ValueError(f"API request failed with status {response.status_code}: {response.get('message', 'Unknown error')}")
            
            # Extract embeddings from response
            if 'output' in response and 'embeddings' in response['output']:
                embeddings_data = response['output']['embeddings']
                logger.debug(f"AlibabaDashScopeEmbedding: Extracted embeddings_data: {str(embeddings_data)[:200]}...") # Added log

                # Extract the actual embedding vectors
                if isinstance(embeddings_data, list) and len(embeddings_data) > 0:
                    # Each item in embeddings_data should have an 'embedding' key
                    first_item = embeddings_data[0]
                    if isinstance(first_item, dict) and 'embedding' in first_item:
                        embedding = first_item['embedding']
                        logger.debug(f"AlibabaDashScopeEmbedding: Successfully extracted embedding vector of length {len(embedding)}") # Added log

                        # Ensure the embedding has the correct dimensions
                        if len(embedding) != self.config.embedding_dims:
                            logger.error(f"AlibabaDashScopeEmbedding: Expected embedding dimension {self.config.embedding_dims}, but got {len(embedding)}") # Added log
                            raise ValueError(
                                f"Expected embedding dimension {self.config.embedding_dims}, "
                                f"but got {len(embedding)}"
                            )
                        
                        return embedding
                    else:
                        # This case might occur if the structure is just a list of vectors directly
                        logger.debug(f"AlibabaDashScopeEmbedding: Embeddings data is a list, returning first item directly (type: {type(first_item)}).") # Added log
                        return first_item
                else:
                    logger.error("AlibabaDashScopeEmbedding: Empty embeddings list in response") # Added log
                    raise ValueError("Empty embeddings list in response")
            else:
                # Debug information
                response_keys = list(response.keys()) if hasattr(response, 'keys') else []
                output_keys = list(response.get('output', {}).keys()) if 'output' in response else []
                logger.error(f"AlibabaDashScopeEmbedding: No embeddings found in response. Response keys: {response_keys}, Output keys: {output_keys}") # Added log
                raise ValueError(f"No embeddings found in response. Response keys: {response_keys}, Output keys: {output_keys}")
                
        except Exception as e:
            logger.error(f"AlibabaDashScopeEmbedding: Error calling DashScope Embedding API: {str(e)}", exc_info=True) # Added log with exc_info
            raise Exception(f"Error calling DashScope Embedding API: {str(e)}")