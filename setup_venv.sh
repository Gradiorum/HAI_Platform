#!/usr/bin/env bash

# Exit on error
set -e

# Name of the virtual environment directory
VENV_DIR=".venv"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "python3 not found! Please install Python 3 before running this script."
    exit 1
fi

echo "Creating a virtual environment in $VENV_DIR ..."
python3 -m venv $VENV_DIR

echo "Activating the virtual environment ..."
source $VENV_DIR/bin/activate

echo "Installing requirements ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete."
echo "To activate it later, run: source $VENV_DIR/bin/activate"
