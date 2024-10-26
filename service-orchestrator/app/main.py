import requests
import time
from fastapi import FastAPI
from requests.exceptions import RequestException

app = FastAPI()

VERTISIM_URL = "http://vertisim_service:5001"

@app.post("/reset_instance")
def reset_instance():
    try:
        # Reset the Vertisim instance
        response_reset = requests.post(f'{VERTISIM_URL}/reset', timeout=120)
        
        if response_reset.status_code != 200:
            raise ConnectionError("Failed to reset Vertisim instance.")

        # Wait for Vertisim to be ready
        wait_for_vertisim()
        
        # print("Requested initial state from Vertisim instance.")
        # Initialize the new Vertisim instance and get initial state
        response_data = requests.get(f'{VERTISIM_URL}/get_initial_state', timeout=120)
        # print("Successfully initialized the new Vertisim instance.")
        
        if response_data.status_code != 200:
            raise ConnectionError(f"Failed to get the initial states from new Vertisim instance. Status code: {response_data.status_code}, Response text: {response_data.text}")
        
        # Return the initial state to the caller
        response_data = response_data.json()
        return {
            "status": "Success", 
            "initial_state": response_data['initial_state'], 
            "action_mask": response_data['action_mask']
        }
            
    except Exception as e:
        return {"status": "Error", "detail": str(e)}
    
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
