#!/bin/bash
# Quick setup script for IMCFlow CI

set -e

# Get the directory where this script is located
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/root/project/tvm"
TEST_DIR="$CI_DIR/.."  # codegen directory
cd "$REPO_DIR"

echo "================================================"
echo "IMCFlow CI Setup"
echo "================================================"
echo ""

# Check if direnv is installed
echo "Checking direnv..."
if command -v direnv &> /dev/null; then
    DIRENV_VERSION=$(direnv version)
    echo "✅ direnv is installed ($DIRENV_VERSION)"
else
    echo "❌ direnv is not installed"
    echo ""
    echo "The CI runner requires direnv to properly load .envrc environment files."
    echo "Please install direnv:"
    echo "  - Debian/Ubuntu: sudo apt-get install direnv"
    echo "  - RHEL/CentOS: sudo yum install direnv"
    echo "  - Or see: https://direnv.net/docs/installation.html"
    echo ""
    exit 1
fi
echo ""

# Check if .envrc files exist and are allowed
echo "Checking .envrc files..."
if [ -f "$REPO_DIR/.envrc" ]; then
    echo "✅ Found: $REPO_DIR/.envrc"
else
    echo "❌ Missing: $REPO_DIR/.envrc"
    exit 1
fi

if [ -f "$REPO_DIR/tvm_practice/.envrc" ]; then
    echo "✅ Found: $REPO_DIR/tvm_practice/.envrc"
else
    echo "❌ Missing: $REPO_DIR/tvm_practice/.envrc"
    exit 1
fi

# Check if .envrc files are allowed (direnv allow)
echo ""
echo "Note: direnv exec will handle .envrc loading automatically"
echo "If you encounter permission issues, you may need to run:"
echo "  cd $REPO_DIR && direnv allow"
echo "  cd $REPO_DIR/tvm_practice && direnv allow"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }
echo "✅ Python 3 found"
echo ""

# Check if pytest is installed in tvm_env
echo "Checking pytest in tvm_env..."
if direnv exec "$REPO_DIR/tvm_practice" python -m pytest --version > /dev/null 2>&1; then
    echo "✅ pytest is installed in tvm_env"
else
    echo "⚠️  pytest not found in tvm_env. Installing..."
    direnv exec "$REPO_DIR/tvm_practice" python -m pip install pytest pytest-json-report
fi

# Check for pytest-json-report specifically
if direnv exec "$REPO_DIR/tvm_practice" python -c "import pytest_jsonreport" > /dev/null 2>&1; then
    echo "✅ pytest-json-report is installed in tvm_env"
else
    echo "⚠️  pytest-json-report not found in tvm_env. Installing..."
    direnv exec "$REPO_DIR/tvm_practice" python -m pip install pytest-json-report
fi
echo ""

# Check if requests library is available (for GitHub API - needed by ci_runner.py)
echo "Checking requests library (for CI runner)..."
if python3 -c "import requests" > /dev/null 2>&1; then
    echo "✅ requests library is installed (system Python)"
else
    echo "⚠️  requests library not found. Installing for system Python..."
    pip install requests
fi
echo ""

# Check if test file exists
echo "Checking test file..."
TEST_FILE="$TEST_DIR/test.py"
if [ -f "$TEST_FILE" ]; then
    echo "✅ Test file found: $TEST_FILE"
else
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi
echo ""

# Create log directory
echo "Creating log directory..."
mkdir -p "$CI_DIR/logs"
echo "✅ Log directory: $CI_DIR/logs"
echo ""

# Check git configuration
echo "Checking git configuration..."
if git remote get-url origin > /dev/null 2>&1; then
    REMOTE=$(git remote get-url origin)
    echo "✅ Git remote: $REMOTE"
else
    echo "❌ Git remote not configured"
    exit 1
fi
echo ""

# Check if on correct branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
if [ "$CURRENT_BRANCH" != "imcflow" ]; then
    echo "⚠️  You're not on the imcflow branch"
    echo "   The CI will monitor the imcflow branch, but you can stay on your current branch"
fi
echo ""

# GitHub token setup
echo "GitHub Token Setup"
echo "------------------"
if [ -n "$GITHUB_TOKEN" ]; then
    echo "✅ GITHUB_TOKEN is set"
    echo "   CI will report status to GitHub"
else
    echo "⚠️  GITHUB_TOKEN is not set"
    echo ""
    echo "To enable GitHub status reporting, you need to set up a GitHub personal access token."
    echo ""
    read -p "Do you want to set up GitHub token now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Please follow these steps:"
        echo "1. Go to: https://github.com/settings/tokens"
        echo "2. Click 'Generate new token (classic)'"
        echo "3. Select scope: 'repo:status'"
        echo "4. Generate and copy the token"
        echo ""
        read -p "Enter your GitHub token: " TOKEN

        if [ -n "$TOKEN" ]; then
            # Save to .ci_env file in CI directory
            cat > "$CI_DIR/.ci_env" << EOF
# IMCFlow CI Environment Variables
export GITHUB_TOKEN="$TOKEN"
EOF
            chmod 600 "$CI_DIR/.ci_env"
            echo "✅ Token saved to $CI_DIR/.ci_env"
            echo ""
            echo "To use the token, run before starting CI:"
            echo "  source $CI_DIR/.ci_env"
            echo ""
            echo "Or add to your ~/.bashrc:"
            echo "  echo 'source $CI_DIR/.ci_env' >> ~/.bashrc"

            # Add to .gitignore
            if ! grep -q "ci/.ci_env" .gitignore 2>/dev/null; then
                echo "ci/.ci_env" >> .gitignore
                echo "✅ Added ci/.ci_env to .gitignore"
            fi
        fi
    else
        echo ""
        echo "Skipping GitHub token setup. You can set it up later by:"
        echo "  export GITHUB_TOKEN='your_token'"
    fi
fi
echo ""

# Make scripts executable
echo "Setting up scripts..."
chmod +x "$CI_DIR/ci_runner.py" "$CI_DIR/ci_manager.sh"
echo "✅ Scripts are executable"
echo ""

# Summary
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Change to the CI directory:"
echo "   cd $CI_DIR"
echo ""
echo "2. Start the CI runner:"
echo "   ./ci_manager.sh start"
echo ""
echo "3. Check status:"
echo "   ./ci_manager.sh status"
echo ""
echo "4. View logs:"
echo "   ./ci_manager.sh logs"
echo ""
echo "5. Read the full guide:"
echo "   cat CI_SETUP_GUIDE.md"
echo ""

if [ -f "$CI_DIR/.ci_env" ]; then
    echo "⚠️  Don't forget to load the environment:"
    echo "   source $CI_DIR/.ci_env"
    echo ""
fi

echo "The CI runner will:"
echo "  • Monitor the 'imcflow' branch every 60 seconds"
echo "  • Run tests on: $TEST_DIR/test.py"
echo "  • Save logs to: $CI_DIR/logs/"
if [ -n "$GITHUB_TOKEN" ]; then
    echo "  • Report status to GitHub"
fi
echo ""
echo "Happy testing! 🚀"
