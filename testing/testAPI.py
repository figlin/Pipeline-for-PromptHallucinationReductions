import requests
import json
import sys

def test_scaledown_api(prompt, context="", model="gpt-4o", rate="0.7"):
    """Test the ScaleDown API with the provided prompt"""
    
    # Try different endpoints
    endpoints = [
       
        "https://api.scaledown.xyz/compress",
      
    ]
    
    # Headers with your API key
    headers = {
        'x-api-key': 'FeMR3ECtaW2FvvgYEtEAG9i9wZCnLa5C9ZEKJN7D',
        'Content-Type': 'application/json'
    }
    
    # Convert rate to appropriate type
    rate_value = float(rate) if rate.replace('.', '').isdigit() else rate
    
    # Payload
    payload = {
        "context": context,
        "prompt": prompt,
        "model": model,
        "scaledown": {
            "rate": rate_value
        }
    }
    
    # Try alternative payload format
    alt_payload = {
        "prompt": prompt,
        "model": model,
        "rate": rate_value
    }
    if context:
        alt_payload["context"] = context
    
    try:
        print(f"Testing ScaleDown API with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print(f"Model: {model}, Rate: {rate}")
        print("-" * 50)
        
        # Try different endpoint and payload combinations
        payloads_to_try = [
            ("Original format", payload),
            ("Alternative format", alt_payload)
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            for payload_name, test_payload in payloads_to_try:
                print(f"  Using {payload_name}: {json.dumps(test_payload, indent=2)}")
                
                response = requests.post(endpoint, headers=headers, data=json.dumps(test_payload))
                
                print(f"  Status Code: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "detail" not in result or "error" not in result.get("detail", "").lower():
                        print(f"  ✓ SUCCESS with {endpoint} using {payload_name}!")
                        print("  Response:")
                        print(json.dumps(result, indent=4))
                        return
                    else:
                        print(f"  ✗ Got error: {result}")
                else:
                    print(f"  ✗ HTTP Error {response.status_code}: {response.text[:100]}")
                print()
        
        print("All endpoints and payload formats failed.")
            
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testAPI.py \"your prompt here\" [context] [model] [rate]")
        print("Example: python testAPI.py \"What is machine learning?\"")
        print("Example: python testAPI.py \"Explain neural networks\" \"AI context\" \"gpt-4o\" \"0\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else ""
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o"
    rate = sys.argv[4] if len(sys.argv) > 4 else "0"
    
    test_scaledown_api(prompt, context, model, rate)