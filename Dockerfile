# Use a lightweight Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install flask transformers torch

# Copy model and inference script
COPY hf_model /model
COPY predictor.py /app/predictor.py

# Set working directory
WORKDIR /app

# Install Gunicorn
RUN pip install gunicorn

# Expose the port
EXPOSE 8080

# Command to run the Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "predictor:app"]
