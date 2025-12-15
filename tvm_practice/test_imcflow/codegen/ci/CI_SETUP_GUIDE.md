# IMCFlow CI Setup Guide

This guide will help you set up automated continuous integration testing for your IMCFlow project that runs on your local machine.

## Overview

The CI system consists of:
1. **ci_runner.py** - Python script that monitors git commits and runs tests
2. **ci_manager.sh** - Shell script to start/stop/manage the CI runner
3. **GitHub Status Integration** - Reports test results back to GitHub (optional)

## Prerequisites

1. Python 3.6+ with pytest
2. Git repository with remote tracking
3. GitHub personal access token (for status reporting)

## Quick Start

### 1. Install Dependencies

```bash
# Install required Python packages
pip install pytest pytest-json-report requests
```

### 2. Set Up GitHub Token (Optional but Recommended)

To enable GitHub commit status updates, you need a personal access token.

**Create a GitHub Token:**
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "TVM CI Runner"
4. Select scopes:
   - `repo:status` (to update commit statuses)
5. Generate token and copy it

**Configure the token:**

Option A - Export in current shell:
```bash
export GITHUB_TOKEN="your_token_here"
```

Option B - Add to your shell profile (~/.bashrc or ~/.zshrc):
```bash
echo 'export GITHUB_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

Option C - Create a .env file (load before starting):
```bash
cat > /root/project/tvm/.ci_env << 'EOF'
export GITHUB_TOKEN="your_token_here"
EOF

# Then source it before starting:
source /root/project/tvm/.ci_env
```

### 3. Start the CI Runner

```bash
cd /root/project/tvm
./ci_manager.sh start
```

This will:
- Start the CI runner in the background
- Begin monitoring the `imcflow` branch for new commits
- Run tests automatically when new commits are detected
- Report results to GitHub (if token is configured)

### 4. Check Status

```bash
./ci_manager.sh status
```

### 5. View Logs

```bash
# View live logs
./ci_manager.sh logs

# View more lines
./ci_manager.sh logs 200

# List all test run logs
./ci_manager.sh list
```

## Usage

### Managing the CI Runner

```bash
# Start the runner
./ci_manager.sh start

# Stop the runner
./ci_manager.sh stop

# Restart the runner
./ci_manager.sh restart

# Check status
./ci_manager.sh status

# View logs (follow mode)
./ci_manager.sh logs

# List all test logs
./ci_manager.sh list
```

### How It Works

1. **Git Polling**: Every 60 seconds, the CI runner:
   - Fetches latest changes from `origin/imcflow`
   - Checks if there's a new commit
   - If yes, triggers a test run

2. **Test Execution**: When a new commit is detected:
   - Sets GitHub status to "pending"
   - Runs pytest on `tvm_practice/test_imcflow/codegen/test.py`
   - Captures all output to a log file
   - Generates JSON report with test statistics

3. **Result Reporting**:
   - Updates GitHub commit status (success/failure)
   - Saves detailed logs to `ci_logs/` directory
   - Preserves test artifacts (JUnit XML, JSON reports)

### Log Files

All logs are saved in `/root/project/tvm/ci_logs/`:
- `ci_runner_main.log` - Main CI runner log
- `test_COMMITHASH_TIMESTAMP.log` - Individual test run logs
- `junit_COMMITHASH_TIMESTAMP.xml` - JUnit XML reports
- `test_COMMITHASH_TIMESTAMP.json` - JSON test reports
- `ci_state.json` - Current CI state (last tested commit)

## Configuration

You can modify the configuration in `ci_runner.py`:

```python
# Change poll interval (default: 60 seconds)
POLL_INTERVAL = 60

# Change branch to monitor (default: imcflow)
BRANCH_NAME = "imcflow"

# Change test file to run
TEST_FILE = REPO_DIR / "tvm_practice/test_imcflow/codegen/test.py"
```

## Testing the Setup

### Manually Trigger a Test

To test without waiting for a new commit:

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
python3 -m pytest test.py -v
```

### Simulate a New Commit

1. Make a small change and commit to imcflow branch
2. Push to remote
3. Watch the CI runner pick it up:
   ```bash
   ./ci_manager.sh logs
   ```

## GitHub Integration

When properly configured, you'll see:

1. **Commit Status Checks** on GitHub:
   - Pending (yellow) - Tests are running
   - Success (green) - All tests passed
   - Failure (red) - Some tests failed
   - Error (red) - CI system error

2. **Status Details**:
   - Number of tests run
   - Number passed/failed
   - Total duration

3. **PR Integration**:
   - Status checks appear on pull requests
   - Can be configured as required checks

## Troubleshooting

### CI Runner Won't Start

```bash
# Check if Python can find pytest
python3 -m pytest --version

# Check if test file exists
ls -la /root/project/tvm/tvm_practice/test_imcflow/codegen/test.py

# Check logs for errors
tail -100 /root/project/tvm/ci_logs/ci_runner_main.log
```

### Tests Not Running

```bash
# Check git fetch works
cd /root/project/tvm
git fetch origin imcflow

# Verify branch exists
git branch -r | grep imcflow

# Check CI state
cat /root/project/tvm/ci_logs/ci_state.json
```

### GitHub Status Not Updating

```bash
# Verify token is set
echo $GITHUB_TOKEN

# Check if requests library is installed
python3 -c "import requests; print('OK')"

# Check logs for GitHub API errors
grep -i "github" /root/project/tvm/ci_logs/ci_runner_main.log
```

### Force Re-run Tests for Current Commit

```bash
# Remove state file to force re-run
rm /root/project/tvm/ci_logs/ci_state.json
./ci_manager.sh restart
```

## Running on System Startup (Optional)

### Using systemd (Recommended for Linux)

Create a systemd service file:

```bash
sudo tee /etc/systemd/system/imcflow-ci.service > /dev/null << 'EOF'
[Unit]
Description=IMCFlow CI Runner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/project/tvm
Environment="GITHUB_TOKEN=your_token_here"
Environment="PYTHONPATH=/root/project/tvm/python"
ExecStart=/usr/bin/python3 /root/project/tvm/ci_runner.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable imcflow-ci.service
sudo systemctl start imcflow-ci.service

# Check status
sudo systemctl status imcflow-ci.service
```

### Using cron (Alternative)

Add to crontab to start on reboot:

```bash
crontab -e

# Add this line:
@reboot cd /root/project/tvm && ./ci_manager.sh start
```

## Advanced Features

### Custom Test Filters

Modify `ci_runner.py` to run only specific tests:

```python
cmd = [
    sys.executable, "-m", "pytest",
    str(self.test_file),
    "-k", "test_one_relu",  # Only run tests matching this pattern
    "-v",
    ...
]
```

### Parallel Test Execution

Install pytest-xdist:
```bash
pip install pytest-xdist
```

Modify command in `ci_runner.py`:
```python
cmd = [
    sys.executable, "-m", "pytest",
    str(self.test_file),
    "-n", "4",  # Run 4 tests in parallel
    ...
]
```

### Notifications

Add email/Slack notifications by modifying `run_tests` method in `ci_runner.py`.

### Web Dashboard

For a web interface to view test results:
```bash
pip install pytest-html
```

Then add to pytest command:
```python
"--html", str(self.log_dir / f"report_{commit_sha[:7]}.html"),
"--self-contained-html"
```

## Best Practices

1. **Monitor disk space**: Test logs can accumulate. Set up log rotation or cleanup:
   ```bash
   # Clean logs older than 30 days
   find /root/project/tvm/ci_logs -name "test_*.log" -mtime +30 -delete
   ```

2. **Keep token secure**: Never commit the token to git
   ```bash
   # Add to .gitignore
   echo ".ci_env" >> .gitignore
   ```

3. **Test locally first**: Always test changes locally before pushing

4. **Check CI status**: Make it a habit to check CI status on GitHub after pushing

## Support

For issues or questions:
1. Check logs: `./ci_manager.sh logs`
2. Verify configuration in `ci_runner.py`
3. Test pytest manually: `python3 -m pytest test.py -v`
4. Check GitHub token permissions
