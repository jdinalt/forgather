#!/usr/bin/env python3
"""
Simple script to test server logging functionality.
"""

import time
import subprocess
import requests
import json

def test_logging():
    print("Testing detailed logging functionality...")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Send a test request
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 30,
        "temperature": 0.7
    }
    
    print("Sending request to server...")
    print(f"Request: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print("\\nCheck server console for detailed logging output!")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_logging()