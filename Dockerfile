FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

# Copy the rest of your application code
COPY --chown=user . .

# Hugging Face Spaces require the app to listen on port 7860
EXPOSE 7860

# Start the FastAPI app using Uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
