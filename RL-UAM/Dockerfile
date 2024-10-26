# Use official Python 3.10 runtime as a parent image. Use the slim version to reduce the image size.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./RL-UAM /app
COPY ./configs /app/configs

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Install GCC and Python development headers
# RUN apt-get update && apt-get install -y gcc python3-dev

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

# # Make port 6006 available to the world outside this container
# EXPOSE 6006

# Run main.py when the container launches
# CMD ["python", "./src/scripts/custom_algo_runner.py", "--config_file", "./src/configs/vertisim_rnd.yaml", "--dataset_dir", "./datasets/rnd"]
CMD ["python", "src/scripts/sb3_runner.py", "--env_config", "/app/configs/config_rl.json", "--rl_config", "/app/configs/sb3/maskedppo.yaml"]
