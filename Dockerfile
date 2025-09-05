# Use official lightweight Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run app with gunicorn (production-ready)
CMD ["gunicorn", "-b", ":8080", "app:app"]
