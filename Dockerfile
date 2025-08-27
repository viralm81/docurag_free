# ==============================
# Step 1: Base Image
# ==============================
FROM python:3.11-slim

# ==============================
# Step 2: Environment setup
# ==============================
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ==============================
# Step 3: Install system dependencies
# ==============================
# ffmpeg is needed by gTTS for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Step 4: Install Python dependencies
# ==============================
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# Step 5: Copy app code
# ==============================
COPY . .

# ==============================
# Step 6: Expose port for Streamlit
# ==============================
EXPOSE 8501

# ==============================
# Step 7: Run the Streamlit app
# ==============================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
