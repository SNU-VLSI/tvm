# IMCFlow Continuous Integration

Automated testing system for the imcflow branch that runs on your local machine.

## Location

This CI system is located in `tvm_practice/test_imcflow/codegen/ci/`

## Quick Start

```bash
# Navigate to the CI directory
cd /root/project/tvm/tvm_practice/test_imcflow/codegen/ci

# 1. Setup (first time only)
./setup_ci.sh

# 2. Start CI runner
./ci_manager.sh start

# 3. Check status
./ci_manager.sh status

# 4. View logs
./ci_manager.sh logs
```

## Commands

```bash
./ci_manager.sh start      # Start CI runner in background
./ci_manager.sh stop       # Stop CI runner
./ci_manager.sh restart    # Restart CI runner
./ci_manager.sh status     # Show current status
./ci_manager.sh logs       # View and follow logs
./ci_manager.sh list       # List all test run logs
```

## How It Works

1. **Monitors** the `imcflow` branch for new commits (every 60 seconds)
2. **Checks out** the specific commit to test (ensuring correct code is tested)
3. **Runs** pytest on `test.py` when changes detected
4. **Uses direnv** to load the correct Python environment from `.envrc` files (tvm_env virtual environment)
5. **Restores** the original branch/state after testing
6. **Reports** results to GitHub as commit statuses (if configured)
7. **Saves** detailed logs to `logs/` directory

## Requirements

- **direnv** - Required to load the correct Python environment from `.envrc` files
- **pytest** and **pytest-json-report** - Test framework and JSON reporting (installed automatically in `tvm_env`)
- Python 3.10+ in the `tvm_env` virtual environment
- Git repository with remote access

## Key Implementation Details

The CI uses `eval "$(direnv export bash)"` instead of `direnv exec` because:

- `direnv exec` loads a different LLVM_PATH that causes TVM library conflicts
- `direnv export` properly sources the `.envrc` hierarchy (codegen → tvm_practice → tvm)
- This matches how developers run tests manually in the directory

## GitHub Integration (Optional)

To enable GitHub status reporting:

1. Create a GitHub personal access token:
   - Go to https://github.com/settings/tokens
   - Generate new token with `repo:status` scope

2. Set the token:
   ```bash
   export GITHUB_TOKEN="your_token_here"
   ```

3. Start the CI runner

Now you'll see test results on GitHub commits and PRs!

## Files

- `ci_runner.py` - Main CI script (monitors git, runs tests, reports to GitHub)
- `ci_manager.sh` - Management script (start/stop/status)
- `setup_ci.sh` - One-time setup script
- `CI_README.md` - This file
- `CI_SETUP_GUIDE.md` - Detailed documentation
- `logs/` - All test logs and reports (created on first run)

## Example Workflow

```bash
# Navigate to CI directory
cd /root/project/tvm/tvm_practice/test_imcflow/codegen/ci

# Initial setup
./setup_ci.sh
source .ci_env  # If you configured GitHub token

# Start monitoring
./ci_manager.sh start

# Make changes to your code (from anywhere in the repo)
cd /root/project/tvm
git checkout imcflow
# ... make changes ...
git commit -m "Add new feature"
git push origin imcflow

# CI automatically detects the new commit and runs tests
# Check what's happening (from ci directory):
cd /root/project/tvm/tvm_practice/test_imcflow/codegen/ci
./ci_manager.sh logs

# View detailed test logs:
ls -lt logs/test_*.log | head -1  # Find latest test log
tail -100 logs/test_<commit>_<timestamp>.log  # View it

# View status on GitHub:
# Go to your commit on GitHub and see the status check!
```

## Understanding Test Results

The CI captures all test outcomes:
- **Passed tests** - All assertions succeeded
- **Failed tests** - Assertion failures or test errors
- **Crashes/Segfaults** - Native code crashes (shown as FAILED with exit code != 0)

Even if tests crash (segfault), the CI will:
- Capture the crash output
- Mark the test as FAILED
- Save the stack trace to the log file
- Report failure status to GitHub (if configured)

## Test Results on GitHub

Once configured, you'll see:

- ⏳ **Pending** - Tests are running
- ✅ **Success** - All tests passed
- ❌ **Failure** - Some tests failed (click "Details" to see which tests failed)
- 🔴 **Error** - CI system error

### Failure Details

When tests fail, the CI will:
1. Post a comment on the commit listing all failed tests
2. The "Details" link in the status will point to that comment
3. You can see exactly which tests failed without checking local logs

Example comment:
```
## ❌ IMCFlow Tests Failed

Commit: 84dece8
Duration: 45.2s

### Summary
- ✅ Passed: 68/120
- ❌ Failed: 52/120

### Failed Tests (52)
1. test_imcflow_model_with_pattern[one_relu-random]
2. test_imcflow_model_with_pattern[one_relu-ones]
...
```

## Logs Location

All logs are in `logs/` (relative to the ci directory):

- `ci_runner_main.log` - CI runner output
- `test_*.log` - Individual test run logs
- `junit_*.xml` - JUnit XML reports
- `test_*.json` - JSON test reports
- `ci_state.json` - Current state

## Troubleshooting

### CI won't start
```bash
# Navigate to CI directory
cd /root/project/tvm/tvm_practice/test_imcflow/codegen/ci

# Check if direnv is installed
which direnv

# Verify .envrc files exist
ls -la ../../../.envrc ../../.envrc ../.envrc

# Check logs
tail -100 logs/ci_runner_main.log

# Verify test file exists
ls -la ../test.py
```

### Wrong Python being used
The CI uses `direnv export` to ensure the correct Python from `tvm_env` is used.
Check the log file to verify:
```bash
# Should show: /root/project/tvm/tvm_practice/tvm_env/bin/python
grep -i "platform linux" logs/ci_runner_main.log
```

### Tests not running
```bash
# Remove state to force re-run
rm logs/ci_state.json
./ci_manager.sh restart
```

### GitHub status not updating
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Check for errors in logs
grep -i github logs/ci_runner_main.log
```

## Advanced Configuration

Edit `ci_runner.py` to customize:

- `POLL_INTERVAL = 60` - How often to check for commits (seconds)
- `BRANCH_NAME = "imcflow"` - Branch to monitor
- `TEST_FILE = ...` - Which test file to run

## For More Information

See [CI_SETUP_GUIDE.md](CI_SETUP_GUIDE.md) for detailed documentation including:
- systemd service setup (auto-start on boot)
- Email/Slack notifications
- Parallel test execution
- Custom test filters
- Log rotation
