# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY inference /app

COPY requirements_common.txt /app/
COPY requirements_serve.txt /app/

# Install necessary dependencies
RUN pip install --no-cache-dir -r requirements_common.txt
RUN pip install --no-cache-dir -r requirements_serve.txt


EXPOSE 8000

# Step 7: Define the command to run the FastAPI app with Uvicorn
CMD ["python", "app.py", "repo_id", "Qishuai/ml_workshop_sample"]
