FROM python:3.10-slim

# Switch to root to install system dependencies
USER root

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face runs as user 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Environment variable for port binding (Hugging Face Spaces expects 7860)
ENV PORT=7860
EXPOSE 7860

# Run Flask server
CMD ["python", "app.py"]
