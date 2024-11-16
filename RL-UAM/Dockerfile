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
CMD ["python", "-W", "ignore", "src/scripts/sb3_runner.py", "--env_config", "/app/config_training.json", "--rl_config", "/app/maskedppo.yaml", "-s"]

# FOR PREDICTION:
# You should comment the above CMD. Then modify the PATH-TO-PREDICTION-CONFIG, PATH-TO-YOUR-TRAINED-ZIP-FILE.ZIP and PATH-TO-YOUR-NORMALIZATION-FILE.PKL to run prediction. You should rebuild docker-compose
# CMD ["python", "-W", "ignore", "src/scripts/prediction.py", "--env_config", "PATH-TO-PREDICTION-CONFIG", "-m", "PATH-TO-YOUR-TRAINED-ZIP-FILE.ZIP", "-n", "PATH-TO-YOUR-NORMALIZATION-FILE.PKL", "-br", "-paxp", "vertisim/vertisim/input/passenger/spill_demand/test"]

