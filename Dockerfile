# Use a base image with Python and other dependencies
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy your Python script and dataset into the container
COPY train_model.py .
COPY serve_model.py .
COPY requirements.txt .
COPY model.joblib .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run your Python script
CMD ["python", "serve_model.py"]
