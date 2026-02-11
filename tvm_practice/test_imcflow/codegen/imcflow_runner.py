"""ImcFlow Runner Abstraction

This module provides an abstraction layer for running ImcFlow tests with different
simulation backends (py_runner and rtl_runner).

Usage:
    runner = get_runner("py")  # or "rtl"
    runner.setup()
    runner.run(binary_name="tvm_host_runner", gdb_mode="no", test_name="my_test", eval_dir="/path/to/eval")
    # Logs are automatically written to eval_dir/logs/runner_name/ during run()
"""

import hashlib
import os
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional


# =============================================================================
# Port Allocator for Concurrent RTL Simulations
# =============================================================================

class PortAllocator:
    """Thread-safe port allocation for concurrent VCS-gem5 simulations.

    Uses file-based locking to allocate unique ports from a specified range.
    """

    PORT_BASE = 10000
    PORT_RANGE = 1000  # Ports 10000-10999
    LOCK_DIR = "/tmp/imcflow_ports"

    @classmethod
    def get_port_for_test(cls, test_name: str, timeout: float = 30.0) -> int:
        """Allocate a unique port for a test based on its name.

        Uses file-based locking to prevent port collisions between
        concurrent simulations.

        Args:
            test_name: Name of the test (used for hash-based port preference)
            timeout: Maximum time to wait for port allocation (seconds)

        Returns:
            Allocated port number

        Raises:
            RuntimeError: If no port could be allocated within timeout
        """
        os.makedirs(cls.LOCK_DIR, exist_ok=True)

        # Hash-based preferred port for deterministic allocation
        hash_val = int(hashlib.md5(test_name.encode()).hexdigest(), 16)
        preferred_port = cls.PORT_BASE + (hash_val % cls.PORT_RANGE)

        start_time = time.time()
        port = preferred_port
        attempts = 0

        while time.time() - start_time < timeout:
            lock_file = os.path.join(cls.LOCK_DIR, f"port_{port}.lock")

            # Try to acquire lock using exclusive file creation
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{test_name}\n{os.getpid()}\n{time.time()}\n".encode())
                os.close(fd)

                # Verify port is actually available for binding
                if cls._is_port_available(port):
                    return port

                # Port not available (maybe used by external process), release lock
                os.unlink(lock_file)

            except FileExistsError:
                # Lock file exists, check if it's stale (older than 2 hours)
                try:
                    mtime = os.path.getmtime(lock_file)
                    if time.time() - mtime > 7200:  # 2 hours
                        os.unlink(lock_file)
                except (OSError, FileNotFoundError):
                    pass

            # Try next port
            port = cls.PORT_BASE + ((port - cls.PORT_BASE + 1) % cls.PORT_RANGE)
            attempts += 1

            if port == preferred_port:
                # All ports tried, wait before retrying
                time.sleep(0.5)

        raise RuntimeError(
            f"Could not allocate port for test '{test_name}' within {timeout}s "
            f"(tried {attempts} ports in range {cls.PORT_BASE}-{cls.PORT_BASE + cls.PORT_RANGE - 1})"
        )

    @classmethod
    def release_port(cls, port: int) -> None:
        """Release a previously allocated port.

        Args:
            port: Port number to release
        """
        lock_file = os.path.join(cls.LOCK_DIR, f"port_{port}.lock")
        try:
            os.unlink(lock_file)
        except FileNotFoundError:
            pass

    @classmethod
    def _is_port_available(cls, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False


# =============================================================================
# Runner Base Class
# =============================================================================

class ImcFlowRunner(ABC):
    """Abstract base class for ImcFlow test runners"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this runner"""
        pass

    @property
    @abstractmethod
    def directory_path(self) -> str:
        """Return the absolute path to the runner directory"""
        pass

    @property
    @abstractmethod
    def timeout(self) -> int:
        """Return the timeout in seconds for this runner"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the display name for this runner (e.g., 'gem5+py_sim Simulation')"""
        pass

    @property
    @abstractmethod
    def main_log_filename(self) -> str:
        """Return the main log filename (e.g., 'gem5.log' or 'gem5_rtl.log')"""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Perform any necessary setup before running (e.g., VCS compilation)"""
        pass

    def run(self, binary_name: str, gdb_mode: str, test_name: str, eval_dir: str) -> None:
        """Run the simulation

        Args:
            binary_name: Name of the binary to execute (e.g., "tvm_host_runner")
            gdb_mode: "yes" or "no" to enable/disable GDB
            test_name: Name of the test for output directory
            eval_dir: Evaluation directory containing the model

        Raises:
            subprocess.CalledProcessError: If simulation fails
        """
        print(f"\n--- Running {self.display_name} ---")

        # Create runner-specific output directory
        output_dir = self.get_output_path(test_name=eval_dir)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        # Create runner-specific log directory in test's logs folder
        runner_log_dir = os.path.join(eval_dir, "logs", self.name)
        os.makedirs(runner_log_dir, exist_ok=True)

        # Convert to absolute path since run.sh executes from runner directory
        abs_runner_log_dir = os.path.abspath(runner_log_dir)

        # Determine imc_size based on IMCFLOW_BIG_IMEM environment variable
        big_imem = os.getenv("IMCFLOW_BIG_IMEM", "").lower() in ("1", "true", "yes")
        imc_size = "270464" if big_imem else "266368"

        # Get SRAM_BACKDOOR setting from environment (default: 1 for performance)
        sram_backdoor = os.getenv("SRAM_BACKDOOR", "1")

        # Prepare environment variables for subprocess
        env = os.environ.copy()
        env["SRAM_BACKDOOR"] = sram_backdoor

        # Pass absolute log directory as 4th argument, imc_size as 5th argument to run.sh
        sim_command = ["direnv", "exec", ".", "./run.sh", binary_name, gdb_mode, test_name, abs_runner_log_dir, imc_size]

        try:
            self._stream_command_output(
                command=sim_command,
                cwd=self.directory_path,
                log_path=os.path.join(runner_log_dir, self.main_log_filename),
                timeout=self.timeout,
                env=env
            )
            print(f"Simulation completed")
            print(f"   Logs saved to: {runner_log_dir}/")
        except subprocess.TimeoutExpired:
            print(f"Simulation timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            raise

    @abstractmethod
    def get_output_path(self, test_name: str) -> str:
        """Get the path to the output file generated by this runner

        Args:
            test_name: Name of the test

        Returns:
            Absolute path to output.npy file
        """
        pass

    def cleanup(self) -> None:
        """Perform any cleanup after running (optional, default is no-op)"""
        pass

    def _stream_command_output(self, command: list, cwd: str, log_path: str,
                                timeout: Optional[int] = None, env: Optional[dict] = None) -> None:
        """Helper to run a command and stream output to both terminal and log file

        Args:
            command: Command and arguments to execute
            cwd: Working directory for the command
            log_path: Path to log file
            timeout: Optional timeout in seconds
            env: Optional environment variables to pass to subprocess

        Raises:
            subprocess.CalledProcessError: If command fails
            subprocess.TimeoutExpired: If command times out
        """
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            # Stream output line by line to both terminal and log file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            except KeyboardInterrupt:
                process.kill()
                process.wait()
                raise

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)


# =============================================================================
# PyRunner Implementation
# =============================================================================

class PyRunner(ImcFlowRunner):
    """Runner for py_runner (Python functional simulation)"""

    @property
    def name(self) -> str:
        return "py_runner"

    @property
    def directory_path(self) -> str:
        return "/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/py_runner"

    @property
    def timeout(self) -> int:
        return 300  # 5 minutes

    @property
    def display_name(self) -> str:
        return "gem5+py_sim Simulation"

    @property
    def main_log_filename(self) -> str:
        return "gem5.log"

    def setup(self) -> None:
        """No setup required for py_runner"""
        print(f"Using {self.name} (Python functional simulation)")
        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"py_runner directory not found at: {self.directory_path}")

    def get_output_path(self, test_name: str) -> str:
        """Get the path to py_runner output file

        Args:
            test_name: Name of the test

        Returns:
            Absolute path to output.npy in py_runner test_outputs
        """
        # With runner_name support, outputs are saved to:
        # {eval_dir}/{test_name}/test_outputs/{runner_name}/output.npy
        eval_dir = "/root/project/tvm/tvm_practice/test_imcflow/codegen"
        return os.path.abspath(
            os.path.join(eval_dir, test_name, "test_outputs", self.name, "output.npy")
        )


# =============================================================================
# RTLRunner Implementation
# =============================================================================

class RTLRunner(ImcFlowRunner):
    """Runner for rtl_runner (VCS RTL co-simulation)"""

    def __init__(self):
        self._allocated_port: Optional[int] = None

    @property
    def name(self) -> str:
        return "rtl_runner"

    @property
    def directory_path(self) -> str:
        return "/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner"

    @property
    def timeout(self) -> int:
        return 7200  # 2 hours

    @property
    def display_name(self) -> str:
        return "gem5+RTL Simulation (may take hours)"

    @property
    def main_log_filename(self) -> str:
        return "gem5_rtl.log"

    def run(self, binary_name: str, gdb_mode: str, test_name: str, eval_dir: str) -> None:
        """Run the RTL simulation with automatic port allocation.

        Allocates a unique socket port for VCS-gem5 communication to enable
        concurrent simulations.
        """
        # Allocate unique port for this test
        self._allocated_port = PortAllocator.get_port_for_test(test_name)
        print(f"Allocated socket port {self._allocated_port} for test '{test_name}'")

        try:
            print(f"\n--- Running {self.display_name} ---")

            # Create runner-specific output directory
            output_dir = self.get_output_path(test_name=eval_dir)
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)

            # Create runner-specific log directory in test's logs folder
            runner_log_dir = os.path.join(eval_dir, "logs", self.name)
            os.makedirs(runner_log_dir, exist_ok=True)

            # Convert to absolute path since run.sh executes from runner directory
            abs_runner_log_dir = os.path.abspath(runner_log_dir)

            # Determine imc_size based on IMCFLOW_BIG_IMEM environment variable
            big_imem = os.getenv("IMCFLOW_BIG_IMEM", "").lower() in ("1", "true", "yes")
            imc_size = 270464 if big_imem else 266368 # 0x42080 or 0x41080
            imc_size += 32
            imc_size = str(imc_size)

            # Build command with socket port as 7th argument
            sim_command = [
                "direnv", "exec", ".", "./run.sh",
                binary_name, gdb_mode, test_name, abs_runner_log_dir, imc_size,
                "",  # npz_dir (empty)
                str(self._allocated_port)
            ]

            self._stream_command_output(
                command=sim_command,
                cwd=self.directory_path,
                log_path=os.path.join(runner_log_dir, self.main_log_filename),
                timeout=self.timeout
            )
            print(f"Simulation completed")
            print(f"   Logs saved to: {runner_log_dir}/")
        except subprocess.TimeoutExpired:
            print(f"Simulation timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            raise
        finally:
            # Release allocated port
            if self._allocated_port is not None:
                PortAllocator.release_port(self._allocated_port)
                print(f"Released socket port {self._allocated_port}")
                self._allocated_port = None

    def setup(self) -> None:
        """Setup for rtl_runner: compile VCS if needed"""
        print(f"Using {self.name} (VCS RTL co-simulation)")

        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"rtl_runner directory not found at: {self.directory_path}")

        # Check if VCS simulator binary exists
        vcs_binary = os.path.join(self.directory_path, "build", "simv_imcflow_gem5")
        if os.path.exists(vcs_binary):
            print(f"VCS simulator found at: {vcs_binary}")
            return

        # Need to compile VCS
        print(f"VCS simulator not found, compiling RTL...")
        compile_command = ["direnv", "exec", ".", "make", "compile"]
        compile_log = os.path.join(self.directory_path, "vcs_compile.log")

        try:
            self._stream_command_output(
                command=compile_command,
                cwd=self.directory_path,
                log_path=compile_log,
                timeout=600  # 10 minutes for compilation
            )
            print(f"VCS compilation successful")
        except subprocess.CalledProcessError as e:
            print(f"VCS compilation failed (exit code {e.returncode})")
            print(f"   Check log at: {compile_log}")
            raise RuntimeError(f"VCS compilation failed. See {compile_log} for details") from e
        except subprocess.TimeoutExpired:
            print(f"VCS compilation timed out after 600 seconds")
            raise RuntimeError("VCS compilation timed out")

    def get_output_path(self, test_name: str) -> str:
        """Get the path to rtl_runner output file

        Args:
            test_name: Name of the test

        Returns:
            Absolute path to output.npy in rtl_runner test_outputs
        """
        # With runner_name support, outputs are saved to:
        # {eval_dir}/{test_name}/test_outputs/{runner_name}/output.npy
        eval_dir = "/root/project/tvm/tvm_practice/test_imcflow/codegen"
        return os.path.abspath(
            os.path.join(eval_dir, test_name, "test_outputs", self.name, "output.npy")
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_runner(runner_type: Optional[str] = None):
    """Factory function to get the appropriate runner(s)

    Args:
        runner_type: Type of runner ("py", "rtl", or "both"). If None, reads from
                     IMCFLOW_RUNNER environment variable (defaults to "py")

    Returns:
        A single ImcFlowRunner instance, or a list of runners if runner_type is "both"

    Raises:
        ValueError: If runner_type is not valid
    """
    if runner_type is None:
        runner_type = os.getenv("IMCFLOW_RUNNER", "py")

    runner_type = runner_type.lower()

    if runner_type == "py":
        return PyRunner()
    elif runner_type == "rtl":
        return RTLRunner()
    elif runner_type == "both":
        return [PyRunner(), RTLRunner()]
    else:
        raise ValueError(
            f"Invalid runner type: '{runner_type}'. "
            f"Valid options are 'py', 'rtl', or 'both'. "
            f"Set via IMCFLOW_RUNNER environment variable or pass to get_runner()"
        )
