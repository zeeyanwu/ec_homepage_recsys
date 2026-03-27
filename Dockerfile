
# 1. Base Image
# Use an official Python runtime as a parent image. We choose slim for a smaller size.
FROM python:3.9-slim

# 2. Set Environment Variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# 3. Set Work Directory
# All subsequent commands will be run from this directory
WORKDIR /app

# 4. Install Dependencies
# Copy the requirements file first to leverage Docker layer caching.
# This means Docker will only re-install dependencies if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Code
# Copy the rest of the application code into the container
COPY . .

# 6. Expose Port
# Inform Docker that the container listens on port 5000 at runtime
# This is the default port for our Flask API server
EXPOSE 8000

# 7. Default Command
# Use Gunicorn as the production-ready WSGI server
# --workers 1 is crucial to avoid multiprocessing issues with libraries like torch/pandas
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "src.serving.recall_api:app"]
