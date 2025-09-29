FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for building libpostal and Python packages
RUN apt-get update && apt-get install -y \
    git \
    autoconf \
    automake \
    libtool \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build and install libpostal from source
RUN git clone https://github.com/openvenues/libpostal /tmp/libpostal && \
    cd /tmp/libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/usr/local/share/libpostal && \
    make -j4 && \
    make install && \
    ldconfig && \
    rm -rf /tmp/libpostal

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    pandas \
    openpyxl \
    torch torchvision torchaudio \
    sentence-transformers \
    scikit-learn \
    rapidfuzz \
    postal \
    tabulate   

# Pre-download HuggingFace model so it's cached in the container
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Copy project files into the container
COPY . /app

# Default command to run the pipeline
CMD ["python3", "main.py"]
