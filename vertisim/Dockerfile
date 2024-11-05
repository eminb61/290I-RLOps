# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config.json \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container to /app
WORKDIR /app

# Install curl and other necessary packages
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire vertisim directory
COPY ./vertisim /app/vertisim
# Copy requirements.txt
COPY ./vertisim/requirements.txt /app/requirements.txt
# Copy config file
COPY ./configs/rl/config_training.json /app/config.json

# Set environment variable for config path
ENV CONFIG_PATH=/app/config.json
ENV PYTHONPATH=/app

# Install Python packages
RUN pip install --no-cache-dir networkx geog pandas
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Add a healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Command to run the application
CMD ["uvicorn", "vertisim.api.main:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "warning"]