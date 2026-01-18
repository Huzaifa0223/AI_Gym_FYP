# Quick Reference - API Improvements

## New Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/models/status` | GET | List all model availability (which are loaded/missing) |
| `/ready` | GET | Readiness probe for k8s/load balancers |
| `/health` | GET | Basic health check |

## Enhanced Response Fields

All `/api/process-frame` responses now include:
```json
{
  // Existing fields (unchanged)
  "success": true,
  "counter": 5,
  "feedback": "✓ GOOD FORM",
  "detected_exercise": "bicep",
  
  // NEW fields
  "degraded": false,        // True if model missing
  "session_id": "uuid...",  // UUID for session tracking
  "api_version": "2.1.0",   // Contract version
  "latency_ms": 45.2        // Processing time
}
```

## Client Integration

### Check Model Availability (startup)
```javascript
const response = await fetch('http://localhost:8000/api/models/status');
const data = await response.json();

// Only enable features where models exist
const availableExercises = data.models
  .filter(m => m.available)
  .map(m => ({ exercise: m.exercise, age: m.age_group }));
```

### Process Frame with Session Continuity
```javascript
// First request - let server generate session_id
const response1 = await fetch('http://localhost:8000/api/process-frame', {
  method: 'POST',
  body: JSON.stringify({
    exercise_type: 'bicep',
    age: 25,
    frame_data: base64Frame,
    session_id: null  // Server generates UUID
  })
});
const data1 = await response1.json();
const sessionId = data1.session_id;  // Store this

// Subsequent requests - reuse session_id
const response2 = await fetch('http://localhost:8000/api/process-frame', {
  method: 'POST',
  body: JSON.stringify({
    exercise_type: 'bicep',
    age: 25,
    frame_data: base64Frame2,
    session_id: sessionId  // Reuse same session
  })
});
```

### Handle Degraded Mode
```javascript
const response = await fetch('http://localhost:8000/api/process-frame', ...);
const data = await response.json();

if (data.degraded) {
  showWarning('⚠️ Limited accuracy - model unavailable for this exercise/age');
}

if (data.model_missing) {
  console.warn('Model not loaded, using heuristics only');
}

if (data.latency_ms > 100) {
  console.warn(`Slow response: ${data.latency_ms}ms`);
}
```

## Testing

```powershell
# Start server
python api_backend.py

# Run test suite
python test_api_improvements.py

# Manual tests
curl http://localhost:8000/api/models/status
curl http://localhost:8000/ready
curl http://localhost:8000/health
```

## Logging Output

```
INFO - Checking model availability...
INFO -   ✅ bicep_adult.pkl
INFO -   ❌ bicep_children.pkl
INFO - Model availability: 1/9 models loaded

INFO - Processed frame: session=abc-123, exercise=bicep, 
       age_group=adult, counter=5, model_missing=False, 
       degraded=False, detection_conf=0.92, latency=45.2ms
```

## Configuration

Edit constants in [api_backend.py](api_backend.py):
```python
MAX_FRAME_SIZE_MB = 5           # Max upload size
MAX_IMAGE_DIMENSION = 1920      # Max width/height
MODELS_DIR = Path("data/models") # Model directory
```
