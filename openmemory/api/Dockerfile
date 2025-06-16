FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /usr/src/openmemory

# Copy the entire project root and install mem0
COPY . /tmp/mem0-project
RUN cd /tmp/mem0-project && pip install -e .

# Copy requirements and install dependencies
COPY openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY openmemory/api/ .
COPY openmemory/api/config.json .

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
