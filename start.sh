#!/bin/bash

# Quick Start Script for mmWave Human Identification Platform
# This script helps you get started with the GUI application

set -e

echo "=========================================="
echo "mmWave ML Platform - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed"
echo "✅ Docker Compose is installed"
echo ""

# Check if data directory exists
if [ ! -d "data/raw" ]; then
    echo "Creating data directories..."
    mkdir -p data/raw data/processed
fi

# Count mesh files
mesh_count=$(find data/raw -type f \( -name "*.ply" -o -name "*.obj" -o -name "*.stl" -o -name "*.off" \) 2>/dev/null | wc -l)

if [ "$mesh_count" -eq 0 ]; then
    echo "⚠️  Warning: No mesh files found in data/raw/"
    echo "Please download FAUST dataset and place .ply or .obj files in data/raw/"
    echo ""
    echo "Download from: http://faust.is.tue.mpg.de/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Found $mesh_count mesh files in data/raw/"
fi

echo ""
echo "=========================================="
echo "Building and starting the application..."
echo "=========================================="
echo ""

# Build and start the container
docker-compose up --build -d

echo ""
echo "=========================================="
echo "✅ Application is starting!"
echo "=========================================="
echo ""
echo "The GUI will be available at:"
echo "👉 http://localhost:8080"
echo ""
echo "Checking if the server is ready..."

# Wait for server to be ready
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        echo ""
        echo "=========================================="
        echo "🚀 Open your browser and go to:"
        echo "   http://localhost:8080"
        echo "=========================================="
        echo ""
        echo "Useful commands:"
        echo "  - View logs:    docker-compose logs -f"
        echo "  - Stop server:  docker-compose down"
        echo "  - Restart:      docker-compose restart"
        echo ""
        exit 0
    fi
    
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

echo ""
echo "⚠️  Server didn't respond in time"
echo "Check logs with: docker-compose logs"
echo ""
echo "You can still try accessing: http://localhost:8080"
