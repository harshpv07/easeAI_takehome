# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Run streamlit when the container launches
CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0"]
