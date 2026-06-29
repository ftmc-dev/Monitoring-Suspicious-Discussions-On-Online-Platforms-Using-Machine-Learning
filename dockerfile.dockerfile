# Use Python 3.11.9 specifically
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]