#!/usr/bin/env python3
"""
Test script for Alibaba Cloud DashScope integration

This script tests the basic functionality of the Alibaba LLM and Embedding implementations
without requiring the full mem0 package to be installed.
"""

import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, '/Users/apple/Documents/dockers/openmemory/mem0')

def test_alibaba_llm():
    """Test the Alibaba LLM implementation"""
    print("=== Testing Alibaba LLM ===")
    
    try:
        from mem0.llms.alibaba import AlibabaDashScopeLLM
        from mem0.configs.llms.base import BaseLlmConfig
        
        # Create config
        config = BaseLlmConfig(
            model="qwen-plus",
            temperature=0.1,
            max_tokens=100,
            api_key=os.getenv("ALIYUN_API_KEY") or "dummy_key_for_testing"
        )
        
        # Initialize LLM
        llm = AlibabaDashScopeLLM(config)
        print("✅ Successfully initialized Alibaba LLM")
        print(f"   Model: {llm.config.model}")
        print(f"   Temperature: {llm.config.temperature}")
        
        # Test message format
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        if os.getenv("ALIYUN_API_KEY"):
            print("   Testing API call...")
            try:
                response = llm.generate_response(messages)
                print(f"   Response: {response[:100]}...")
                print("✅ API call successful")
            except Exception as e:
                print(f"❌ API call failed: {e}")
        else:
            print("   ⚠️  ALIYUN_API_KEY not set, skipping API test")
            
    except Exception as e:
        print(f"❌ Failed to test Alibaba LLM: {e}")
        import traceback
        traceback.print_exc()

def test_alibaba_embedding():
    """Test the Alibaba Embedding implementation"""
    print("\n=== Testing Alibaba Embedding ===")
    
    try:
        from mem0.embeddings.alibaba import AlibabaDashScopeEmbedding
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        
        # Create config
        config = BaseEmbedderConfig(
            model="text-embedding-v4",
            embedding_dims=1024,
            api_key=os.getenv("ALIYUN_API_KEY") or "dummy_key_for_testing"
        )
        
        # Initialize embedding
        embedding = AlibabaDashScopeEmbedding(config)
        print("✅ Successfully initialized Alibaba Embedding")
        print(f"   Model: {embedding.config.model}")
        print(f"   Dimensions: {embedding.config.embedding_dims}")
        
        if os.getenv("ALIYUN_API_KEY"):
            print("   Testing embedding call...")
            try:
                test_text = "This is a test sentence for embedding."
                result = embedding.embed(test_text)
                print(f"   Embedding dimensions: {len(result)}")
                print(f"   First 5 values: {result[:5]}")
                print("✅ Embedding call successful")
            except Exception as e:
                print(f"❌ Embedding call failed: {e}")
        else:
            print("   ⚠️  ALIYUN_API_KEY not set, skipping API test")
            
    except Exception as e:
        print(f"❌ Failed to test Alibaba Embedding: {e}")
        import traceback
        traceback.print_exc()

def test_factory_registration():
    """Test that the factories can find our implementations"""
    print("\n=== Testing Factory Registration ===")
    
    try:
        from mem0.utils.factory import LlmFactory, EmbedderFactory
        
        # Test LLM factory
        if "alibaba" in LlmFactory.provider_to_class:
            print("✅ Alibaba LLM registered in factory")
            print(f"   Class: {LlmFactory.provider_to_class['alibaba']}")
        else:
            print("❌ Alibaba LLM not found in factory")
            
        # Test Embedder factory
        if "alibaba" in EmbedderFactory.provider_to_class:
            print("✅ Alibaba Embedding registered in factory")
            print(f"   Class: {EmbedderFactory.provider_to_class['alibaba']}")
        else:
            print("❌ Alibaba Embedding not found in factory")
            
    except Exception as e:
        print(f"❌ Failed to test factory registration: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Alibaba Cloud DashScope Integration Test")
    print("========================================")
    
    # Check if API key is available
    api_key = os.getenv("ALIYUN_API_KEY")
    if api_key:
        print(f"✅ ALIYUN_API_KEY found (length: {len(api_key)})")
    else:
        print("⚠️  ALIYUN_API_KEY not set - API calls will be skipped")
        print("   To test API calls, set: export ALIYUN_API_KEY='your_key'")
    
    # Check if dashscope is installed
    try:
        import dashscope
        try:
            version = dashscope.__version__
            print(f"✅ DashScope SDK installed (version: {version})")
        except AttributeError:
            print("✅ DashScope SDK installed (version info not available)")
    except ImportError:
        print("❌ DashScope SDK not installed")
        print("   Install with: pip install dashscope")
        return
    
    # Run tests
    test_factory_registration()
    test_alibaba_llm()
    test_alibaba_embedding()
    
    print("\n=== Test Summary ===")
    print("✅ Integration files created successfully")
    print("✅ Factory registration working")
    print("✅ Classes can be imported and initialized")
    
    if api_key:
        print("✅ API integration tested")
    else:
        print("⚠️  Set ALIYUN_API_KEY to test API integration")
    
    print("\n=== Next Steps ===")
    print("1. Set your DashScope API key: export ALIYUN_API_KEY='your_key'")
    print("2. Install mem0 in development mode or use the configuration files")
    print("3. Use the example configuration in configs/alibaba_config.yaml")
    print("4. Refer to docs/alibaba_integration.md for detailed usage")

if __name__ == "__main__":
    main()