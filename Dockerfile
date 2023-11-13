# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir fastapi langchain langserve uvicorn openai sse_starlette

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run the application
CMD ["uvicorn", "langchain_serve:app", "--host", "0.0.0.0", "--port", "8000"]
