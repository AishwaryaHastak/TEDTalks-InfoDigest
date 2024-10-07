# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/
COPY setup.py /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . /app/

# Install the local package (without -e option)
RUN pip install --no-cache-dir /app

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]
