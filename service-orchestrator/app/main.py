import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests.exceptions import RequestException
from typing import Dict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VERTISIM_URL = "http://vertisim_service:5001"

# Track active instances
active_instances: Dict[str, str] = {}

def ensure_vertisim_ready():
    """Ensure Vertisim service is up and ready"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f'{VERTISIM_URL}/health', timeout=30)
            if response.status_code == 200:
                return True
        except RequestException:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise HTTPException(status_code=503, detail="Service Orchestrator is live but Vertisim service is not available")
    return False

@app.post("/create_instance")
async def create_instance():
    """Create a new Vertisim instance"""
    ensure_vertisim_ready()
    
    try:
        # Create new instance
        response = requests.post(f'{VERTISIM_URL}/create_instance', timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                              detail="Failed to create Vertisim instance")
        
        instance_data = response.json()
        instance_id = instance_data["instance_id"]
        active_instances[instance_id] = "active"
        
        return {"instance_id": instance_id, "status": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/instance/{instance_id}")
async def delete_instance(instance_id: str):
    """Delete a Vertisim instance"""
    try:
        response = requests.delete(f'{VERTISIM_URL}/instance/{instance_id}', timeout=60)
        if response.status_code == 200:
            active_instances.pop(instance_id, None)
            return {"status": "Success"}
        raise HTTPException(status_code=response.status_code, 
                          detail="Failed to delete instance")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_instance")
async def reset_instance():
    """Create and initialize a new instance"""
    try:
        # Create new instance
        create_response = await create_instance()
        instance_id = create_response["instance_id"]
        
        # Reset the instance
        reset_response = requests.post(f'{VERTISIM_URL}/instance/{instance_id}/reset', timeout=60)
        if reset_response.status_code != 200:
            raise HTTPException(status_code=reset_response.status_code, 
                              detail="Failed to reset instance")
        
        # Get initial state
        init_state_response = requests.get(
            f'{VERTISIM_URL}/instance/{instance_id}/initial_state', 
            timeout=60
        )
        if init_state_response.status_code != 200:
            raise HTTPException(status_code=init_state_response.status_code,
                              detail="Failed to get initial state")
        
        response_data = init_state_response.json()
        return {
            "status": "Success",
            "instance_id": instance_id,
            "initial_state": response_data['initial_state'],
            "action_mask": response_data['action_mask']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        ensure_vertisim_ready()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    
def wait_for_vertisim(timeout: int = 120):
    """
    Checks whether Vertisim is ready to accept the next request.
    """
    start_time = time.time()

    while True:
        try:
            response = requests.get(f'{VERTISIM_URL}/status', timeout=60)
            if response.status_code == 200:
                break
        except RequestException as e:
            print(f"Waiting for Vertisim to be ready. Error: {str(e)}")
            time.sleep(1) # Delay for 1 second before next attempt

        # Check if timeout has been reached
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out while waiting for Vertisim to be ready.")
