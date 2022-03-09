# lightweight python
FROM python:3.7-slim

RUN apt-get update

# Copy local code to the container image.
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./datasets/ /app/datasets/
COPY ./notebooks/ /app/notebooks/
COPY ./src/ /app/src/
COPY ./tests/ /app/tests/

# Run the jupyter lab on container startup
CMD [ "jupyter", "lab", "--allow-root", "--port=8888", "--ip=0.0.0.0" ]