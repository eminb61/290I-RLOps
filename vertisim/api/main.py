from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Any, Union, Dict
import sys
import numpy as np
import uuid
sys.path.append('/app')

from vertisim.vertisim.instance_manager import InstanceManager
from .models import create_action_models
from .config import CONFIG  # Import the loaded configuration

class Actions(BaseModel):
    actions: List[int]
    # actions: int

app = FastAPI()

# Dictionary to store multiple instances
instances: Dict[str, InstanceManager] = {}

# instance_manager = InstanceManager(config=CONFIG)

def get_instance_manager(instance_id: str) -> InstanceManager:
    """Get instance manager by ID or raise exception if not found"""
    if instance_id not in instances:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    return instances[instance_id]


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    # If all instances are ready, return healthy
    if all(instance.status for instance in instances.values()):
            return {"status": "Success", "message": "All VertiSim instances are ready."}


@app.post("/create_instance")
def create_instance():
    """Create a new Vertisim instance"""
    try:
        instance_id = str(uuid.uuid4())
        instances[instance_id] = InstanceManager(config=CONFIG)
        return {"instance_id": instance_id, "status": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.delete("/instance/{instance_id}")
def delete_instance(instance_id: str):
    """Delete a Vertisim instance"""
    if instance_id in instances:
        instances[instance_id].close()
        del instances[instance_id]
    return {"status": "Success"}


@app.get("/instance/{instance_id}/status")
def get_status(instance_id: str):
    instance = get_instance_manager(instance_id)
    if not instance.status:
        raise HTTPException(status_code=503, detail="VertiSim is not ready.")
    return {"status": "Success", "message": "VertiSim is ready."}


@app.post("/instance/{instance_id}/reset")
def reset(instance_id: str):
    try:
        instance = get_instance_manager(instance_id)
        result = instance.reset()
        return {"status": "Success", "message": "Vertisim instance reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/instance/{instance_id}/initial_state")
def get_initial_state(instance_id: str):
    try:
        instance = get_instance_manager(instance_id)
        initial_state = instance.get_initial_state()
        return convert_numpy_types(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instance/{instance_id}/step")
def step(instance_id: str, actions: Actions):
    try:
        instance = get_instance_manager(instance_id)
        if not instance.status:
            raise HTTPException(status_code=503, detail="Vertisim is not ready.")
        
        new_state, reward, terminated, truncated, action_mask = instance.step(actions.actions)
        
        response = {
            "new_state": convert_numpy_types(new_state),
            "reward": float(reward) if isinstance(reward, np.number) else reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_mask": convert_numpy_types(action_mask)
        }
        
        return response
    except Exception as e:
        print(f"Error in step endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instance/{instance_id}/space_params")
def get_space_params(instance_id: str):
    instance = get_instance_manager(instance_id)
    if instance.sim_instance is None:
        raise HTTPException(
            status_code=500, 
            detail="Simulation instance not initialized. Please reset first."
        )
    
    try:
        params = {
            "n_actions": instance.sim_instance.get_action_count(),
            "n_aircraft": instance.sim_instance.get_aircraft_count(),
            "n_vertiports": instance.sim_instance.get_vertiport_count(),
            "n_vertiport_state_variables": instance.sim_instance.get_vertiport_state_variable_count(),
            "n_aircraft_state_variables": instance.sim_instance.get_aircraft_state_variable_count(),
            "n_environmental_state_variables": instance.sim_instance.get_environmental_state_variable_count(),
            "n_additional_state_variables": instance.sim_instance.get_additional_state_variable_count()
        }
        return convert_numpy_types(params)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting space parameters: {str(e)}"
        )


# @app.get("/get_space_params")
# def get_space_params():
#     if instance_manager.sim_instance is None:
#         raise HTTPException(
#             status_code=500, 
#             detail="Simulation instance not initialized. Please reset first."
#         )
    
#     try:
#         return {
#             "n_actions": instance_manager.sim_instance.get_action_count(),
#             "n_aircraft": instance_manager.sim_instance.get_aircraft_count(),
#             "n_vertiports": instance_manager.sim_instance.get_vertiport_count(),
#             "n_vertiport_state_variables": instance_manager.sim_instance.get_vertiport_state_variable_count(),
#             "n_aircraft_state_variables": instance_manager.sim_instance.get_aircraft_state_variable_count(),
#             "n_environmental_state_variables": instance_manager.sim_instance.get_environmental_state_variable_count(),
#             "n_additional_state_variables": instance_manager.sim_instance.get_additional_state_variable_count()
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error getting space parameters: {str(e)}"
#         )


# @app.get("/get_space_params")
# def get_space_params():
#     if instance_manager.sim_instance is None:
#         raise HTTPException(
#             status_code=500, 
#             detail="Simulation instance not initialized. Please reset first."
#         )
    
#     try:
#         params = {
#             "n_actions": instance_manager.sim_instance.get_action_count(),
#             "n_aircraft": instance_manager.sim_instance.get_aircraft_count(),
#             "n_vertiports": instance_manager.sim_instance.get_vertiport_count(),
#             "n_vertiport_state_variables": instance_manager.sim_instance.get_vertiport_state_variable_count(),
#             "n_aircraft_state_variables": instance_manager.sim_instance.get_aircraft_state_variable_count(),
#             "n_environmental_state_variables": instance_manager.sim_instance.get_environmental_state_variable_count(),
#             "n_additional_state_variables": instance_manager.sim_instance.get_additional_state_variable_count()
#         }
#         return convert_numpy_types(params)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error getting space parameters: {str(e)}"
#         )


@app.get("/instance/{instance_id}/performance_metrics")
def get_performance_metrics(instance_id: str):
    instance = get_instance_manager(instance_id)
    if instance.sim_instance is None:
        raise HTTPException(
            status_code=500, 
            detail="Simulation instance not initialized. Please reset first."
        )
    
    try:
        return instance.get_performance_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting performance metrics: {str(e)}"
        )

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001) 

