# Project Name - Machine Learning MLOps Pipeline

## Docker Containerization

This section explains how to containerize your machine learning model and its dependencies using Docker.

### Prerequisites

Before you proceed, make sure you have Docker installed on your local machine.

### Dockerfile

In this project, we use a Dockerfile to define the environment and dependencies for running the machine learning model. Here's the Dockerfile:

```Dockerfile
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
```
The Dockerfile specifies the base image, sets the working directory, copies your Python script and dataset into the container, installs necessary Python dependencies, and defines the command to run your Python script.

### Building the Docker Image
To build the Docker image, execute the following command in your project directory where the Dockerfile is located:
```bash
docker build -t lept0n5/mlops:1.0 .
```
Replace your-image-name with the desired name for your Docker image.

### Running the Docker Container
Once the Docker image is built, you can run the container with the following command:
```bash
docker run lept0n5/mlops:1.0
```

### Pushing the Docker Image to a Registry
To make your Docker image available for deployment on a cloud platform, you can push it to a container registry like Docker Hub. Here are the steps:

1. Log in to Docker Hub (create an account if you don't have one):
```bash
docker login
```
2. Tag your image with your Docker Hub username and image name:
```bash
docker tag lept0n5/mlops:1.0 lept0n5/mlops:1.0
```
3. Push the image to Docker Hub:
```bash
docker push lept0n5/mlops:1.0 lept0n5/mlops:1.0
```
Now your Docker image is available on Docker Hub for deployment.# MLOPs
