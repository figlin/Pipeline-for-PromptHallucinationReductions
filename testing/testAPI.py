import requests
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_scaledown_api(prompt, context="", model="gpt-4o", rate="0.7"):
    """Test the ScaleDown API with the provided prompt"""
    
    # Get API key from environment or use default
    api_key = os.getenv('SCALEDOWN_API_KEY', 'wN63sz8ykq1JFycUpAOnE5PN3AXaPqoZ6g71oMcN')
    
    # Based on ScaleDown docs, use the correct endpoint
    endpoint = "https://api.scaledown.xyz/compress"
    
    # Headers with your API key
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    print(f"Using API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print(f"Testing ScaleDown API with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"Model: {model}, Rate: {rate}, Context: '{context[:30]}{'...' if len(context) > 30 else ''}'")
    print("=" * 60)
    
    # Convert rate to appropriate type
    try:
        rate_value = float(rate) if isinstance(rate, str) and rate.replace('.', '').isdigit() else rate
    except ValueError:
        rate_value = 0
        print(f"Warning: Invalid rate '{rate}', using default 0.7")
    
    # Based on ScaleDown docs, the correct payload format
    payload = {
        "prompt": prompt,
        "model": model,
        "context": context,
        "scaledown": {
            "rate": rate_value
        }
    }
    
    print(f"Testing endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            endpoint, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"SUCCESS!")
                print("Response:")
                print(json.dumps(result, indent=4))
                
                # Check for common response fields
                response_fields = list(result.keys())
                print(f"Response fields: {response_fields}")
                
                # Look for compressed/response text
                for field in ['compressed_prompt', 'compressed', 'text', 'output', 'response', 'full_response']:
                    if field in result and result[field]:
                        print(f"Found {field}: {result[field][:100]}{'...' if len(str(result[field])) > 100 else ''}")
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response.text[:200]}")
                
        elif response.status_code == 401:
            print(f"Authentication failed - check your API key")
            print(f"Response: {response.text}")
            
        elif response.status_code == 400:
            print(f"Bad request - check payload format")
            print(f"Response: {response.text}")
            
        elif response.status_code == 404:
            print(f"Endpoint not found")
            
        elif response.status_code == 429:
            print(f"Rate limited")
            print(f"Response: {response.text}")
            
        else:
            print(f"HTTP Error {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"Timeout error")
    except requests.exceptions.ConnectionError:
        print(f"Connection error - check network")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_basic_connectivity():
    """Test basic connectivity to ScaleDown"""
    print("Testing basic connectivity...")
    
    try:
        response = requests.get("https://api.scaledown.xyz/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("ScaleDown API is reachable")
        else:
            print(f"ScaleDown API returned {response.status_code}")
    except Exception as e:
        print(f"Cannot reach ScaleDown API: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testAPI.py \"your prompt here\" [context] [model] [rate]")
        print("Example: python testAPI.py \"What is machine learning?\"")
        print("Example: python testAPI.py \"Explain neural networks\" \"AI context\" \"gpt-4o\" \"0.7\"")
        print("\nOr run with --connectivity to test basic API connectivity")
        sys.exit(1)
    
    if sys.argv[1] == "--connectivity":
        test_basic_connectivity()
        sys.exit(0)
    
    prompt = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else ""
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o"
    rate = sys.argv[4] if len(sys.argv) > 4 else "auto"
    
    # Test connectivity first
    test_basic_connectivity()
    print()
    
    # Test API
    test_scaledown_api(prompt, context, model, rate)