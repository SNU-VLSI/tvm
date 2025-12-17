#!/usr/bin/env python3
"""
IMCFlow Continuous Integration Runner

This script:
1. Monitors the imcflow branch for new commits
2. Runs pytest on test_imcflow/codegen/test.py when changes detected
3. Reports test results back to GitHub as commit statuses
4. Runs tests in background with progress tracking
"""

import os
import sys
import time
import subprocess
import json
import datetime
import signal
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Configuration
REPO_DIR = Path("/root/project/tvm")
CI_DIR = Path(__file__).parent.resolve()  # codegen/ci directory
TEST_FILE = CI_DIR.parent / "test.py"  # codegen/test.py
BRANCH_NAME = "imcflow"
POLL_INTERVAL = 10  # seconds between git checks
LOG_DIR = CI_DIR / "logs"
STATE_FILE = LOG_DIR / "ci_state.json"

# GitHub API configuration
# Set these environment variables or create a config file
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = "SNU-VLSI/tvm"  # owner/repo format

class CIRunner:
    def __init__(self):
        self.repo_dir = REPO_DIR
        self.test_file = TEST_FILE
        self.branch = BRANCH_NAME
        self.log_dir = LOG_DIR
        self.log_dir.mkdir(exist_ok=True)

        self.current_commit = None
        self.last_tested_commit = self.load_state()
        self.test_process = None
        self.running = True
        self.original_branch = None  # Track original branch to restore after testing

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def load_state(self) -> Optional[str]:
        """Load the last tested commit from state file"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    return state.get("last_tested_commit")
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
        return None

    def save_state(self, commit_sha: str):
        """Save the current commit as last tested"""
        state = {
            "last_tested_commit": commit_sha,
            "last_test_time": datetime.datetime.now().isoformat()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def get_current_commit(self) -> Optional[str]:
        """Get the current commit SHA of the branch"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"origin/{self.branch}"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting current commit: {e}")
            return None

    def fetch_updates(self) -> bool:
        """Fetch latest changes from remote"""
        try:
            subprocess.run(
                ["git", "fetch", "origin", self.branch],
                cwd=self.repo_dir,
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error fetching updates: {e}")
            return False

    def get_commit_info(self, commit_sha: str) -> Dict[str, str]:
        """Get commit message and author"""
        try:
            result = subprocess.run(
                ["git", "show", "-s", "--format=%s|%an|%ae", commit_sha],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            parts = result.stdout.strip().split('|')
            return {
                "message": parts[0] if len(parts) > 0 else "",
                "author": parts[1] if len(parts) > 1 else "",
                "email": parts[2] if len(parts) > 2 else ""
            }
        except subprocess.CalledProcessError:
            return {"message": "", "author": "", "email": ""}

    def get_current_branch(self) -> Optional[str]:
        """Get the current branch name or HEAD if detached"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting current branch: {e}")
            return None

    def checkout_commit(self, commit_sha: str) -> bool:
        """Checkout to a specific commit

        Args:
            commit_sha: The commit SHA to checkout

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save the original branch if not already saved
            if self.original_branch is None:
                self.original_branch = self.get_current_branch()
                print(f"📌 Saved original branch/state: {self.original_branch}")

            # Checkout the commit
            print(f"🔄 Checking out commit {commit_sha[:7]}...")
            result = subprocess.run(
                ["git", "checkout", commit_sha],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Successfully checked out {commit_sha[:7]}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking out commit {commit_sha}: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return False

    def restore_original_state(self) -> bool:
        """Restore the repository to its original branch/state

        Returns:
            True if successful, False otherwise
        """
        if self.original_branch is None:
            print("⚠️  No original branch to restore")
            return True

        try:
            print(f"🔄 Restoring to original branch: {self.original_branch}")
            result = subprocess.run(
                ["git", "checkout", self.original_branch],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Successfully restored to {self.original_branch}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error restoring original branch: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return False

    def set_github_status(self, commit_sha: str, state: str, description: str,
                          context: str = "ci/imcflow-tests", target_url: str = None):
        """Set GitHub commit status using GitHub API

        Args:
            commit_sha: The commit to update
            state: pending, success, error, or failure
            description: Short description of the status
            context: Status context (shown in GitHub UI)
            target_url: Optional URL for "Details" link
        """
        if not GITHUB_TOKEN:
            print("Warning: GITHUB_TOKEN not set, skipping GitHub status update")
            return

        url = f"https://api.github.com/repos/{GITHUB_REPO}/statuses/{commit_sha}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        payload = {
            "state": state,
            "description": description,
            "context": context
        }

        # Add target_url if provided
        if target_url:
            payload["target_url"] = target_url

        try:
            import requests
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 201:
                print(f"✅ GitHub status updated: {state} - {description}")
            else:
                print(f"⚠️  Failed to update GitHub status: {response.status_code} - {response.text}")
        except ImportError:
            print("Warning: 'requests' library not available, using curl instead")
            self._set_github_status_curl(url, headers, payload)
        except Exception as e:
            print(f"Error updating GitHub status: {e}")

    def _set_github_status_curl(self, url: str, headers: Dict, payload: Dict):
        """Fallback method using curl"""
        try:
            subprocess.run([
                "curl", "-X", "POST",
                "-H", f"Authorization: {headers['Authorization']}",
                "-H", f"Accept: {headers['Accept']}",
                "-d", json.dumps(payload),
                url
            ], check=True, capture_output=True)
            print(f"✅ GitHub status updated via curl")
        except Exception as e:
            print(f"Error updating GitHub status via curl: {e}")

    def post_github_comment(self, commit_sha: str, body: str) -> Optional[str]:
        """Post a comment on the commit with test results

        Args:
            commit_sha: The commit to comment on
            body: Comment text (markdown supported)

        Returns:
            URL of the comment, or None if failed
        """
        if not GITHUB_TOKEN:
            return None

        url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{commit_sha}/comments"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        payload = {"body": body}

        try:
            import requests
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 201:
                comment_url = response.json().get("html_url")
                print(f"✅ GitHub comment posted: {comment_url}")
                return comment_url
            else:
                print(f"⚠️  Failed to post GitHub comment: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error posting GitHub comment: {e}")
            return None

    def run_tests(self, commit_sha: str) -> Tuple[bool, str, Dict]:
        """Run pytest on the test file

        Returns:
            (success, log_path, test_stats)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"test_{commit_sha[:7]}_{timestamp}.log"
        json_report = self.log_dir / f"test_{commit_sha[:7]}_{timestamp}.json"

        print(f"\n{'='*60}")
        print(f"Running tests for commit {commit_sha[:7]}")
        print(f"Log file: {log_file}")
        print(f"{'='*60}\n")

        # Set GitHub status to pending
        self.set_github_status(commit_sha, "pending", "Running IMCFlow tests...")

        # Checkout the commit to test
        if not self.checkout_commit(commit_sha):
            error_msg = f"Failed to checkout commit {commit_sha[:7]}"
            print(f"❌ {error_msg}")
            self.set_github_status(commit_sha, "error", error_msg)
            return False, str(log_file), {
                "total": 0, "passed": 0, "failed": 0,
                "skipped": 0, "duration": 0, "failed_tests": []
            }

        # Prepare test command
        # We need to run the test file directly (not via pytest) because:
        # 1. test.py uses tvm.testing.main() which handles pytest internally
        # 2. direnv exec causes LLVM library conflicts that lead to segfaults
        # 3. Running directly in the directory with sourced .envrc works correctly

        # Build the command to:
        # 1. cd to test directory
        # 2. source .envrc (via direnv allow + manual sourcing)
        # 3. run python test.py
        test_cmd = f"""
cd {self.test_file.parent} && \
eval "$(direnv export bash)" && \
python test.py --verbose -k "random" --tb=short \
  --junit-xml {self.log_dir / f"junit_{commit_sha[:7]}_{timestamp}.xml"} \
  --json-report \
  --json-report-file {json_report} \
  --json-report-indent 2
"""

        cmd = ["bash", "-c", test_cmd]

        # Run tests and capture output
        # Use try-finally to ensure we restore the original state
        try:
            start_time = time.time()
            with open(log_file, 'w') as f:
                f.write(f"IMCFlow CI Test Run\n")
                f.write(f"Commit: {commit_sha}\n")
                f.write(f"Branch: {self.branch}\n")
                f.write(f"Started: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Test directory: {self.test_file.parent}\n")
                f.write(f"Using direnv export to load .envrc environment\n")
                f.write(f"{'='*60}\n\n")
                f.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                self.test_process = process

                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()

                process.wait()

            end_time = time.time()
            duration = end_time - start_time

            # Parse test results
            test_stats = self._parse_test_results(json_report, duration)
            success = process.returncode == 0

            # Update GitHub status and post comment with details
            if success:
                description = f"All tests passed ({test_stats['total']} tests, {test_stats['duration']:.1f}s)"
                self.set_github_status(commit_sha, "success", description)
            else:
                description = f"Tests failed ({test_stats['failed']}/{test_stats['total']} failed)"

                # Create a comment with failed test details
                comment_body = self._create_failure_comment(commit_sha, test_stats, str(log_file))
                comment_url = self.post_github_comment(commit_sha, comment_body)

                # Set status with link to comment
                self.set_github_status(commit_sha, "failure", description, target_url=comment_url)

            # Write summary to log
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Test Summary:\n")
                f.write(f"  Total: {test_stats['total']}\n")
                f.write(f"  Passed: {test_stats['passed']}\n")
                f.write(f"  Failed: {test_stats['failed']}\n")
                f.write(f"  Duration: {test_stats['duration']:.1f}s\n")
                f.write(f"  Status: {'PASSED' if success else 'FAILED'}\n")
                f.write(f"{'='*60}\n")

            return success, str(log_file), test_stats

        finally:
            # Always restore the original branch/state after testing
            print(f"\n{'='*60}")
            self.restore_original_state()
            print(f"{'='*60}\n")

    def _create_failure_comment(self, commit_sha: str, test_stats: Dict, log_file: str) -> str:
        """Create a markdown comment body for test failures

        Args:
            commit_sha: The commit being tested
            test_stats: Test statistics dict
            log_file: Path to the log file

        Returns:
            Markdown formatted comment body
        """
        failed_tests = test_stats.get("failed_tests", [])
        total = test_stats.get("total", 0)
        failed = test_stats.get("failed", 0)
        passed = test_stats.get("passed", 0)
        duration = test_stats.get("duration", 0)

        # Build comment body
        lines = [
            f"## ❌ IMCFlow Tests Failed",
            f"",
            f"**Commit:** `{commit_sha[:7]}`",
            f"**Duration:** {duration:.1f}s",
            f"",
            f"### Summary",
            f"- ✅ **Passed:** {passed}/{total}",
            f"- ❌ **Failed:** {failed}/{total}",
            f"",
        ]

        if failed_tests:
            lines.append(f"### Failed Tests ({len(failed_tests)})")
            lines.append("")

            # Limit to first 50 failed tests to avoid huge comments
            display_tests = failed_tests[:50]
            for i, test_name in enumerate(display_tests, 1):
                lines.append(f"{i}. `{test_name}`")

            if len(failed_tests) > 50:
                lines.append("")
                lines.append(f"... and {len(failed_tests) - 50} more failed tests")

        lines.append("")
        lines.append(f"---")
        lines.append(f"*Local log file: `{log_file}`*")
        lines.append(f"")
        lines.append(f"<sub>Generated by IMCFlow CI</sub>")

        return "\n".join(lines)

    def _parse_test_results(self, json_report: Path, duration: float) -> Dict:
        """Parse pytest JSON report for test statistics"""
        stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": duration,
            "failed_tests": []  # List of failed test names
        }

        if not json_report.exists():
            return stats

        try:
            with open(json_report) as f:
                data = json.load(f)
                summary = data.get("summary", {})
                stats["total"] = summary.get("total", 0)
                stats["passed"] = summary.get("passed", 0)
                stats["failed"] = summary.get("failed", 0)
                stats["skipped"] = summary.get("skipped", 0)

                # Extract failed test names
                tests = data.get("tests", [])
                for test in tests:
                    if test.get("outcome") == "failed":
                        test_name = test.get("nodeid", "unknown")
                        # Clean up the nodeid (remove file path prefix)
                        if "::" in test_name:
                            test_name = test_name.split("::")[-1]
                        stats["failed_tests"].append(test_name)

        except Exception as e:
            print(f"Warning: Could not parse test results: {e}")

        return stats

    def check_and_run(self, force=False):
        """Check for new commits and run tests if needed

        Args:
            force: If True, run tests even if commit hasn't changed
        """
        # Fetch latest changes
        if not self.fetch_updates():
            return

        # Get current commit
        commit_sha = self.get_current_commit()
        if not commit_sha:
            return

        # Check if this is a new commit
        if not force and commit_sha == self.last_tested_commit:
            return

        # New commit detected (or forced run)!
        commit_info = self.get_commit_info(commit_sha)
        if force and commit_sha == self.last_tested_commit:
            print(f"\n🔄 Running tests on current commit (startup)")
        else:
            print(f"\n🆕 New commit detected!")
        print(f"   SHA: {commit_sha[:7]}")
        print(f"   Message: {commit_info['message']}")
        print(f"   Author: {commit_info['author']} <{commit_info['email']}>")

        # Run tests
        try:
            success, log_path, stats = self.run_tests(commit_sha)

            # Save state
            self.save_state(commit_sha)
            self.last_tested_commit = commit_sha

            # Print summary
            status_icon = "✅" if success else "❌"
            print(f"\n{status_icon} Tests {'PASSED' if success else 'FAILED'}")
            print(f"   Total: {stats['total']}, Passed: {stats['passed']}, Failed: {stats['failed']}")
            print(f"   Duration: {stats['duration']:.1f}s")
            print(f"   Log: {log_path}\n")

        except Exception as e:
            print(f"❌ Error running tests: {e}")
            self.set_github_status(commit_sha, "error", f"CI error: {str(e)}")

    def run_loop(self):
        """Main loop that polls for changes"""
        print("🚀 IMCFlow CI Runner Started")
        print(f"   Repository: {self.repo_dir}")
        print(f"   Branch: {self.branch}")
        print(f"   Test file: {self.test_file}")
        print(f"   Poll interval: {POLL_INTERVAL}s")
        print(f"   Log directory: {self.log_dir}")
        if GITHUB_TOKEN:
            print(f"   GitHub reporting: Enabled")
        else:
            print(f"   GitHub reporting: Disabled (set GITHUB_TOKEN to enable)")
        print(f"\nRunning initial test on startup...\n")

        # Run tests immediately on startup (force=True to run even if commit unchanged)
        try:
            self.check_and_run(force=True)
        except Exception as e:
            print(f"Error in initial test run: {e}")

        print(f"\nWaiting for new commits...\n")

        while self.running:
            try:
                self.check_and_run()
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                self.shutdown(None, None)
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(POLL_INTERVAL)

    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        print("\n\n🛑 Shutting down CI runner...")
        self.running = False

        # Kill test process if running
        if self.test_process and self.test_process.poll() is None:
            print("   Terminating running test process...")
            self.test_process.terminate()
            try:
                self.test_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("   Force killing test process...")
                self.test_process.kill()

        print("   Goodbye!")
        sys.exit(0)


def main():
    # Check if test file exists
    if not TEST_FILE.exists():
        print(f"❌ Error: Test file not found: {TEST_FILE}")
        sys.exit(1)

    # Check if in correct directory
    if not (REPO_DIR / ".git").exists():
        print(f"❌ Error: Not a git repository: {REPO_DIR}")
        sys.exit(1)

    runner = CIRunner()
    runner.run_loop()


if __name__ == "__main__":
    main()
