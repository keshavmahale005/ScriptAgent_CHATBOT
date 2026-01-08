#!/bin/bash

echo "=================================================="
echo "Retell AI Clone - Setup Script"
echo "=================================================="
echo ""

# Check if Ollama is installed
echo "Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama is already installed"
fi

# Start Ollama service
echo ""
echo "Starting Ollama service..."
ollama serve &
sleep 5

# Download recommended model
echo ""
echo "Downloading Qwen 2.5:7b model (this may take a few minutes)..."
ollama pull qwen2.5:7b

echo ""
echo "✅ Ollama model downloaded!"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To start the application, run:"
echo "  streamlit run app.py"
echo ""