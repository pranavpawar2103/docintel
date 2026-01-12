"""Test that all dependencies are properly installed."""

import pytest
from src.utils.config import settings


def test_imports():
    """Test that all required packages can be imported."""
    import openai
    import langchain
    import chromadb
    import fastapi
    import streamlit
    assert True


def test_api_key_loaded():
    """Test that OpenAI API key is loaded."""
    assert settings.openai_api_key.startswith("sk-")


def test_openai_api():
    """Test OpenAI API connection."""
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API works!'"}],
        max_tokens=10
    )
    
    assert "API works" in response.choices[0].message.content


def test_chromadb():
    """Test ChromaDB initialization."""
    import chromadb
    client = chromadb.Client()
    
    # Create test collection
    collection = client.create_collection("test")
    assert collection.name == "test"
    
    # Clean up
    client.delete_collection("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])