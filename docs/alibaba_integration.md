# Alibaba Cloud DashScope Integration for Mem0

This guide explains how to use Alibaba Cloud's DashScope API with Mem0, including the qwen-plus LLM and text-embedding-v4 embedding model.

## Prerequisites

1. **Alibaba Cloud Account**: You need an active Alibaba Cloud account
2. **DashScope API Key**: Get your API key from the [DashScope Console](https://dashscope.console.aliyun.com/)
3. **Python Dependencies**: Install the required package

## Installation

### 1. Install DashScope SDK

```bash
pip install dashscope
```

### 2. Set Environment Variable

```bash
export ALIYUN_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```env
ALIYUN_API_KEY=your_api_key_here
```

## Configuration

### Basic Configuration

```python
from mem0 import Memory

config = {
    "llm": {
        "provider": "alibaba",
        "config": {
            "model": "qwen-plus",
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.8,
            "api_key": "your_api_key_here"  # or use env:ALIYUN_API_KEY
        }
    },
    "embedder": {
        "provider": "alibaba",
        "config": {
            "model": "text-embedding-v4",
            "embedding_dims": 1024,
            "api_key": "your_api_key_here"  # or use env:ALIYUN_API_KEY
        }
    }
}

memory = Memory.from_config(config)
```

### YAML Configuration

Create a `config.yaml` file:

```yaml
llm:
  provider: alibaba
  config:
    model: qwen-plus
    temperature: 0.1
    max_tokens: 2000
    top_p: 0.8
    api_key: env:ALIYUN_API_KEY

embedder:
  provider: alibaba
  config:
    model: text-embedding-v4
    embedding_dims: 1024
    api_key: env:ALIYUN_API_KEY
```

Then load it:

```python
from mem0 import Memory

memory = Memory.from_config("config.yaml")
```

## Available Models

### LLM Models

- `qwen-plus`: Latest and most capable model (recommended)
- `qwen-turbo`: Faster inference, good for most tasks
- `qwen-max`: Maximum capability model
- `qwen-long`: Optimized for long context

### Embedding Models

- `text-embedding-v4`: Latest embedding model (1024 dimensions, recommended)
- `text-embedding-v3`: Previous generation (1536 dimensions)
- `text-embedding-v2`: Older model (1536 dimensions)

## Usage Examples

### Basic Usage

```python
import os
from mem0 import Memory

# Set up configuration
config = {
    "llm": {
        "provider": "alibaba",
        "config": {
            "model": "qwen-plus",
            "api_key": os.getenv("ALIYUN_API_KEY")
        }
    },
    "embedder": {
        "provider": "alibaba",
        "config": {
            "model": "text-embedding-v4",
            "api_key": os.getenv("ALIYUN_API_KEY")
        }
    }
}

# Initialize memory
memory = Memory.from_config(config)

# Add memories
memory.add("I love Chinese cuisine", user_id="user_001")
memory.add("I work as a software engineer", user_id="user_001")

# Search memories
results = memory.search("What does the user like to eat?", user_id="user_001")
print(results)

# Get all memories
all_memories = memory.get_all(user_id="user_001")
print(all_memories)
```

### Advanced Configuration

```python
config = {
    "llm": {
        "provider": "alibaba",
        "config": {
            "model": "qwen-plus",
            "temperature": 0.2,  # Control randomness (0.0 - 1.0)
            "max_tokens": 4000,  # Maximum response length
            "top_p": 0.9,       # Nucleus sampling parameter
            "api_key": os.getenv("ALIYUN_API_KEY")
        }
    },
    "embedder": {
        "provider": "alibaba",
        "config": {
            "model": "text-embedding-v4",
            "embedding_dims": 1024,
            "api_key": os.getenv("ALIYUN_API_KEY")
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "my_memories",
            "embedding_model_dims": 1024  # Must match embedding dimensions
        }
    }
}
```

## Running the Example

1. Set your API key:
   ```bash
   export ALIYUN_API_KEY="your_api_key_here"
   ```

2. Run the example script:
   ```bash
   python examples/alibaba_example.py
   ```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you have installed dashscope:
   ```bash
   pip install dashscope
   ```

2. **API Key Error**: Ensure your API key is correctly set:
   ```bash
   echo $ALIYUN_API_KEY
   ```

3. **Model Not Found**: Check that you're using a valid model name from the available models list.

4. **Dimension Mismatch**: Ensure your vector store configuration matches the embedding dimensions:
   - `text-embedding-v4`: 1024 dimensions
   - `text-embedding-v3`: 1536 dimensions

### Getting Help

- [DashScope Documentation](https://help.aliyun.com/zh/dashscope/)
- [Alibaba Cloud Console](https://dashscope.console.aliyun.com/)
- [Mem0 Documentation](https://docs.mem0.ai/)

## Performance Tips

1. **Model Selection**:
   - Use `qwen-turbo` for faster responses
   - Use `qwen-plus` for better quality
   - Use `qwen-max` for maximum capability

2. **Embedding Optimization**:
   - `text-embedding-v4` offers the best performance with 1024 dimensions
   - Consider batch processing for multiple texts

3. **Cost Optimization**:
   - Monitor your API usage in the DashScope console
   - Use appropriate model sizes for your use case
   - Cache embeddings when possible

## Security Notes

- Never hardcode API keys in your source code
- Use environment variables or secure configuration management
- Regularly rotate your API keys
- Monitor API usage for unexpected activity