# Use a base image with Python and other dependencies
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy your Python script and dataset into the container
COPY train_model.py .

# Install Python dependencies
RUN pip install scikit-learn pandas

# Command to run your Python script
CMD ["python", "train_model.py"]
