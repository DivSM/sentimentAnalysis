FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to be accessed by the outside world
EXPOSE 5000

ENV FLASK_ENV=development

CMD ["python", "app.py"]