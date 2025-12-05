# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=1

# Install system dependencies - FIXED: use libgl1 instead of libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime dependencies
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \ 
    # Build dependencies (only needed during installation)
    build-essential \
    gcc \
    # Cleanup to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY . .

# Create directories for models
RUN mkdir -p models/lstm_predictor models/vision_transformer models/gnn_sensor_network models/ensemble

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]