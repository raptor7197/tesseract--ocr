#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# Setup script for Natural Scene Text Detection & Recognition
#
# This script:
#   1. Installs system dependencies (Tesseract OCR)
#   2. Creates a Python virtual environment (optional)
#   3. Installs Python dependencies from requirements.txt
#   4. Downloads the pre-trained EAST text detection model
#   5. Creates necessary output directories
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh            # full setup with venv
#   ./setup.sh --no-venv  # skip virtual environment creation
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

USE_VENV=true
VENV_DIR="venv"
EAST_MODEL_DIR="models"
EAST_MODEL_FILE="frozen_east_text_detection.pb"
EAST_MODEL_PATH="${EAST_MODEL_DIR}/${EAST_MODEL_FILE}"
EAST_MODEL_URL="https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"

# ── Parse arguments ──────────────────────────────────────────────────────────

for arg in "$@"; do
    case $arg in
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-venv    Skip Python virtual environment creation"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Run './setup.sh --help' for usage information."
            exit 1
            ;;
    esac
done

# ── Helper functions ─────────────────────────────────────────────────────────

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ── Banner ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Natural Scene Text Detection & Recognition — Setup    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python ─────────────────────────────────────────────────────

info "Checking Python installation..."

PYTHON_CMD=""
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    fail "Python 3 is not installed. Please install Python 3.8+ and try again."
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    fail "Python 3.8+ is required, but found $PYTHON_VERSION"
fi

success "Python $PYTHON_VERSION found ($PYTHON_CMD)"

# ── Step 2: Install system dependencies ──────────────────────────────────────

info "Checking system dependencies..."

install_tesseract() {
    if command_exists apt-get; then
        info "Installing Tesseract OCR via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-eng
    elif command_exists dnf; then
        info "Installing Tesseract OCR via dnf..."
        sudo dnf install -y tesseract tesseract-langpack-eng
    elif command_exists yum; then
        info "Installing Tesseract OCR via yum..."
        sudo yum install -y tesseract tesseract-langpack-eng
    elif command_exists pacman; then
        info "Installing Tesseract OCR via pacman..."
        sudo pacman -Sy --noconfirm tesseract tesseract-data-eng
    elif command_exists brew; then
        info "Installing Tesseract OCR via Homebrew..."
        brew install tesseract
    else
        fail "Could not detect package manager. Please install Tesseract OCR manually:
    Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng
    Fedora:        sudo dnf install tesseract tesseract-langpack-eng
    Arch:          sudo pacman -S tesseract tesseract-data-eng
    macOS:         brew install tesseract"
    fi
}

if command_exists tesseract; then
    TESS_VERSION=$(tesseract --version 2>&1 | head -1)
    success "Tesseract already installed: $TESS_VERSION"
else
    warn "Tesseract OCR not found. Attempting to install..."
    install_tesseract

    if command_exists tesseract; then
        TESS_VERSION=$(tesseract --version 2>&1 | head -1)
        success "Tesseract installed: $TESS_VERSION"
    else
        fail "Tesseract installation failed. Please install it manually."
    fi
fi

# ── Step 3: Set up Python virtual environment ────────────────────────────────

PIP_CMD="pip3"

if [ "$USE_VENV" = true ]; then
    info "Setting up Python virtual environment..."

    if [ -d "$VENV_DIR" ]; then
        warn "Virtual environment already exists at ./$VENV_DIR"
    else
        $PYTHON_CMD -m venv "$VENV_DIR" || fail "Failed to create virtual environment. Install python3-venv:
    sudo apt install python3-venv"
        success "Created virtual environment at ./$VENV_DIR"
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
    PIP_CMD="pip"
    success "Activated virtual environment"
else
    info "Skipping virtual environment (--no-venv)"
fi

# ── Step 4: Install Python dependencies ──────────────────────────────────────

info "Installing Python dependencies..."

if [ ! -f "requirements.txt" ]; then
    fail "requirements.txt not found in $SCRIPT_DIR"
fi

$PIP_CMD install --upgrade pip -q
$PIP_CMD install -r requirements.txt -q

success "Python dependencies installed"

# Also install dev/test dependencies if available
if [ -f "requirements-dev.txt" ]; then
    info "Installing development dependencies..."
    $PIP_CMD install -r requirements-dev.txt -q
    success "Development dependencies installed"
fi

# ── Step 5: Download EAST model ──────────────────────────────────────────────

info "Checking EAST text detection model..."

mkdir -p "$EAST_MODEL_DIR"

if [ -f "$EAST_MODEL_PATH" ]; then
    FILE_SIZE=$(stat -f%z "$EAST_MODEL_PATH" 2>/dev/null || stat -c%s "$EAST_MODEL_PATH" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -gt 90000000 ]; then
        success "EAST model already downloaded ($EAST_MODEL_PATH, ~$(( FILE_SIZE / 1048576 ))MB)"
    else
        warn "EAST model file exists but seems too small (${FILE_SIZE} bytes). Re-downloading..."
        rm -f "$EAST_MODEL_PATH"
    fi
fi

if [ ! -f "$EAST_MODEL_PATH" ]; then
    info "Downloading EAST model (~96 MB)..."

    if command_exists wget; then
        wget -q --show-progress -O "$EAST_MODEL_PATH" "$EAST_MODEL_URL" || {
            rm -f "$EAST_MODEL_PATH"
            fail "Failed to download EAST model via wget"
        }
    elif command_exists curl; then
        curl -L --progress-bar -o "$EAST_MODEL_PATH" "$EAST_MODEL_URL" || {
            rm -f "$EAST_MODEL_PATH"
            fail "Failed to download EAST model via curl"
        }
    else
        fail "Neither wget nor curl found. Please download the EAST model manually:
    URL:  $EAST_MODEL_URL
    Save: $EAST_MODEL_PATH"
    fi

    FILE_SIZE=$(stat -f%z "$EAST_MODEL_PATH" 2>/dev/null || stat -c%s "$EAST_MODEL_PATH" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -lt 90000000 ]; then
        rm -f "$EAST_MODEL_PATH"
        fail "Downloaded file seems too small (${FILE_SIZE} bytes). The download may have failed."
    fi

    success "EAST model downloaded to $EAST_MODEL_PATH (~$(( FILE_SIZE / 1048576 ))MB)"
fi

# ── Step 6: Create output directories ────────────────────────────────────────

info "Creating output directories..."

mkdir -p output/annotated
mkdir -p output/results
mkdir -p data/sample_images
mkdir -p data/ground_truth

success "Output directories created"

# ── Step 7: Verify installation ──────────────────────────────────────────────

info "Verifying installation..."

VERIFY_RESULT=$($PYTHON_CMD -c "
import sys
errors = []

try:
    import cv2
except ImportError:
    errors.append('opencv-python')

try:
    import numpy
except ImportError:
    errors.append('numpy')

try:
    import pytesseract
except ImportError:
    errors.append('pytesseract')

try:
    import imutils
except ImportError:
    errors.append('imutils')

try:
    from PIL import Image
except ImportError:
    errors.append('Pillow')

if errors:
    print('MISSING:' + ','.join(errors))
    sys.exit(1)
else:
    print('OK')
    sys.exit(0)
" 2>&1) || true

if echo "$VERIFY_RESULT" | grep -q "^OK$"; then
    success "All core Python packages verified"
else
    MISSING=$(echo "$VERIFY_RESULT" | grep "MISSING:" | sed 's/MISSING://')
    warn "Some packages may not be properly installed: $MISSING"
    warn "Try running: pip install $MISSING"
fi

# Verify Tesseract can be called from Python
TESS_CHECK=$($PYTHON_CMD -c "
import pytesseract
try:
    v = pytesseract.get_tesseract_version()
    print(f'OK:{v}')
except Exception as e:
    print(f'FAIL:{e}')
" 2>&1) || true

if echo "$TESS_CHECK" | grep -q "^OK:"; then
    TESS_PY_VER=$(echo "$TESS_CHECK" | sed 's/OK://')
    success "Tesseract accessible from Python (version: $TESS_PY_VER)"
else
    warn "Tesseract may not be accessible from Python. Check your PATH."
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Setup Complete!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$USE_VENV" = true ]; then
    echo -e "To activate the virtual environment in a new terminal:"
    echo -e "  ${YELLOW}source ${VENV_DIR}/bin/activate${NC}"
    echo ""
fi

echo -e "Quick start commands:"
echo ""
echo -e "  ${YELLOW}# Process a single image${NC}"
echo -e "  python main.py --input data/sample_images/sign.jpg --output output/"
echo ""
echo -e "  ${YELLOW}# Batch process a directory${NC}"
echo -e "  python main.py --input data/sample_images/ --output output/ --batch"
echo ""
echo -e "  ${YELLOW}# Launch the web UI${NC}"
echo -e "  streamlit run app.py"
echo ""
echo -e "  ${YELLOW}# Run tests${NC}"
echo -e "  python -m pytest tests/ -v"
echo ""
