# Use official Python 3.11 runtime as a parent image. Use the slim version to reduce the image size.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Install GCC and Python development headers
# RUN apt-get update && apt-get install -y gcc python3-dev

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006

# Run main.py when the container launches
CMD ["python", "-W", "ignore", "src/scripts/prediction.py", "--env_config", "/app/config_rl_prediction.json", "-m" "/app/model/MaskablePPO_2024-11-04_23-27-21.zip", "-n", "MaskablePPO_2024-11-04_23-27-21_vecnormalize.pkl", "-br", "paxp", "/app/vertisim/input/passenger/spill_demand/test"]
# CMD ["python", "-W", "ignore", "src/scripts/sb3_runner.py", "--env_config", "/app/config_rl_training.json", "--rl_config", "/app/maskedppo.yaml", "-s", "-n", "1"]
