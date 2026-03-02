"""
Simple endpoint tester - tests each endpoint individually
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url, method='GET'):
    """Test a single endpoint"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        if method == 'GET':
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, timeout=5)
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to {url}")
        print("Make sure server is running: python api_backend.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting API Endpoint Tests...")
    print(f"Target: {BASE_URL}\n")
    
    tests = [
        ("Root Endpoint", f"{BASE_URL}/"),
        ("Health Check", f"{BASE_URL}/health"),
        ("Readiness Check", f"{BASE_URL}/ready"),
        ("Models Status", f"{BASE_URL}/api/models/status"),
        ("Exercises List", f"{BASE_URL}/api/exercises"),
        ("Age Groups", f"{BASE_URL}/api/age-groups"),
    ]
    
    results = []
    for name, url in tests:
        result = test_endpoint(name, url)
        results.append((name, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        sys.exit(1)
