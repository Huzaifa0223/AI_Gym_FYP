"""
Quick test script for API improvements
Tests: models/status, /ready, /health, and UUID sessions
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_models_status():
    """Test the new models status endpoint"""
    print("\n=== Testing /api/models/status ===")
    response = requests.get(f"{BASE_URL}/api/models/status")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"Summary: {data['summary']}")
    print(f"\nModel Details:")
    for model in data['models']:
        status = "✅" if model['available'] else "❌"
        print(f"  {status} {model['model_name']}")
    
    return response.status_code == 200

def test_readiness():
    """Test the readiness endpoint"""
    print("\n=== Testing /ready ===")
    response = requests.get(f"{BASE_URL}/ready")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"Ready: {data['ready']}")
    print(f"MediaPipe Initialized: {data['mediapipe_initialized']}")
    print(f"Models Available: {data['models_available']}/{data['models_total']}")
    
    return response.status_code == 200

def test_health():
    """Test the health endpoint"""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"Status: {data['status']}")
    print(f"Active Sessions: {data['active_sessions']}")
    
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("\n=== Testing / ===")
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"Service: {data['service']}")
    print(f"Version: {data['version']}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        results = []
        results.append(("Root Endpoint", test_root()))
        results.append(("Health Check", test_health()))
        results.append(("Readiness Check", test_readiness()))
        results.append(("Models Status", test_models_status()))
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {name}")
        
        all_passed = all(r[1] for r in results)
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running: python api_backend.py")
