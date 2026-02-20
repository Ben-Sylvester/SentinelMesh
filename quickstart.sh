#!/bin/bash
# SentinelMesh v3.0 - Quick Start Script
# One-command deployment for local development and testing

set -e

echo "════════════════════════════════════════════════════════════"
echo "🚀 SentinelMesh v3.0 - Quick Start"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Python 3.10+ required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $python_version${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/memory data/cache data/workflows data/prompts
echo -e "${GREEN}✅ Data directories created${NC}"

# Setup configuration
echo ""
if [ ! -f ".env" ]; then
    echo "Setting up configuration..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file and add your API keys${NC}"
    echo ""
    echo "Required API keys:"
    echo "  - OPENAI_API_KEY (for core features)"
    echo "  - ANTHROPIC_API_KEY (optional)"
    echo "  - GOOGLE_API_KEY (optional)"
    echo ""
    read -p "Press Enter to open .env file in editor..."
    ${EDITOR:-nano} .env
else
    echo -e "${GREEN}✅ Configuration file exists${NC}"
fi

# Check if API keys are set
echo ""
echo "Validating configuration..."
source .env
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY not set in .env${NC}"
    echo "Please edit .env and add your OpenAI API key"
    exit 1
fi
echo -e "${GREEN}✅ Configuration valid${NC}"

# Initialize database
echo ""
echo "Initializing database..."
if [ -f "migrate_database.py" ]; then
    python migrate_database.py
    echo -e "${GREEN}✅ Database initialized${NC}"
fi

# Start server
echo ""
echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}🎉 SentinelMesh is ready!${NC}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Starting server on http://localhost:8000"
echo ""
echo "Available endpoints:"
echo "  • API:       http://localhost:8000/docs"
echo "  • Health:    http://localhost:8000/health"
echo "  • Dashboard: http://localhost:8501 (if enabled)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start with hot-reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8000
