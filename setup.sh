#!/bin/bash

# Update package lists and install system dependencies needed by Playwright
apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2

# Install the Playwright browser executable
# The --with-deps flag installs necessary OS dependencies for Chromium
playwright install --with-deps chromium

# Run your Streamlit app
# Make sure the path to your Python script is correct
streamlit run src/RAG_DIALOGUE_SWISSCOM.py