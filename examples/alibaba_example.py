#!/usr/bin/env python3
"""
Alibaba Cloud DashScope Example for Mem0

This example demonstrates how to use Alibaba Cloud's qwen-plus LLM 
and text-embedding-v4 embedding model with Mem0.

Prerequisites:
1. Install dashscope: pip install dashscope
2. Set your DashScope API key: export ALIYUN_API_KEY="your_api_key_here"
3. Get your API key from: https://dashscope.console.aliyun.com/
"""

import os
from mem0 import Memory

def main():
    # Check if API key is set
    if not os.getenv("ALIYUN_API_KEY"):
        print("Error: Please set ALIYUN_API_KEY environment variable")
        print("You can get your API key from: https://dashscope.console.aliyun.com/")
        return

    # Configuration for Alibaba Cloud DashScope
    config = {
        "llm": {
            "provider": "alibaba",
            "config": {
                "model": "qwen-plus",
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.8,
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
                "collection_name": "mem0_alibaba_demo",
                "embedding_model_dims": 1024
            }
        }
    }

    # Initialize Mem0 with Alibaba Cloud configuration
    print("Initializing Mem0 with Alibaba Cloud DashScope...")
    try:
        memory = Memory.from_config(config)
        print("✅ Successfully initialized Mem0 with Alibaba Cloud!")
    except Exception as e:
        print(f"❌ Failed to initialize Mem0: {e}")
        return

    # Example usage
    user_id = "alibaba_user_001"
    
    print("\n=== Adding memories ===")
    
    # Add some memories
    memories_to_add = [
        "I love Chinese cuisine, especially Sichuan food.",
        "I work as a software engineer in Beijing.",
        "My favorite programming language is Python.",
        "I enjoy reading science fiction novels in my free time.",
        "I'm planning to visit Shanghai next month."
    ]
    
    for text in memories_to_add:
        try:
            result = memory.add(text, user_id=user_id)
            print(f"✅ Added: {text}")
            print(f"   Memory ID: {result}")
        except Exception as e:
            print(f"❌ Failed to add memory: {e}")
    
    print("\n=== Searching memories ===")
    
    # Search for memories
    search_queries = [
        "What food does the user like?",
        "Where does the user work?",
        "What are the user's hobbies?"
    ]
    
    for query in search_queries:
        try:
            results = memory.search(query, user_id=user_id)
            print(f"\n🔍 Query: {query}")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['memory']} (Score: {result.get('score', 'N/A')})")
        except Exception as e:
            print(f"❌ Failed to search: {e}")
    
    print("\n=== Getting all memories ===")
    
    # Get all memories for the user
    try:
        all_memories = memory.get_all(user_id=user_id)
        print(f"Total memories for user {user_id}: {len(all_memories)}")
        for i, mem in enumerate(all_memories, 1):
            print(f"   {i}. {mem['memory']}")
    except Exception as e:
        print(f"❌ Failed to get memories: {e}")
    
    print("\n=== Demo completed! ===")
    print("\nTip: You can modify the configuration in this script to experiment with different models:")
    print("- LLM models: qwen-plus, qwen-turbo, qwen-max")
    print("- Embedding models: text-embedding-v4, text-embedding-v3")

if __name__ == "__main__":
    main()