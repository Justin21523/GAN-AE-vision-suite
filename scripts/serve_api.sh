#!/bin/bash

# GAN-AE-VISION-SUITE API Server Startup Script

set -e

echo "🎨 Starting GAN-AE-VISION-SUITE API Server"

# Check if AI_CACHE_ROOT is set
if [ -z "$AI_CACHE_ROOT" ]; then
    echo "❌ AI_CACHE_ROOT environment variable is not set"
    echo "Please set AI_CACHE_ROOT to your AI warehouse directory"
    exit 1
fi

echo "📁 AI Warehouse: $AI_CACHE_ROOT"

# Create necessary directories
mkdir -p $AI_CACHE_ROOT/{models,outputs,train,logs,metrics}

# Set Python path
export PYTHONPATH=src

# Start API server
echo "🚀 Starting FastAPI server on 0.0.0.0:8000"
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1