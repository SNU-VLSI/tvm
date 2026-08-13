"""ImcFlow Runner Abstraction

This module provides an abstraction layer for running ImcFlow tests with different
simulation backends (py_runner and rtl_runner).

Usage:
    runner = get_runner("py")  # or "rtl"
    runner.setup()
    runner.run(binary_name="execute_graph", gdb_mode="no", test_name="my_test", eval_dir="/path/to/eval")
    # Logs are automatically written to eval_dir/logs/runner_name/ during run()
"""

import hashlib
import os
import shutil
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

    def run(self, binary_name: str, gdb_mode: str, test_name: str, eval_dir: str,
            noise_csv: Optional[str] = None,
            noise_layout_json: Optional[str] = None,
            noise_mode: Optional[str] = None,
            noise_table_format: Optional[str] = None,
            noise_granularity: Optional[str] = None,
            noise_seed: Optional[int] = None,
            sample_idx: Optional[int] = None) -> None:
        """Run the simulation

        Args:
            binary_name: Name of the binary to execute (e.g., "execute_graph")
            gdb_mode: "yes" or "no" to enable/disable GDB
            test_name: Name of the test for output directory
            eval_dir: Evaluation directory containing the model
            noise_csv: Optional path to the ADC noise CSV. Forwarded to
                py_runner's run.sh as the 6th positional arg, which gem5's
                run_imcflow.py turns into --noise-csv → IMCFLOW_NOISE_CSV.
                Ignored by RTL runner (RTL doesn't run the Python IMCU model).
            noise_layout_json: Optional path to imce_map noise layout JSON
                (concat_per_core.json). Forwarded as the 7th positional arg;
                run_imcflow.py exports IMCFLOW_NOISE_LAYOUT_JSON for IMCU.
            noise_mode: Optional ADC noise sampling mode. 'sample'/'alias'
                (default empirical) or 'greedy' (argmax over diff_bin).
                Forwarded as the 8th positional arg; run_imcflow.py exports
                IMCFLOW_NOISE_MODE for IMCU.
            noise_table_format: Optional table format ('auto', 'wpattern_ref',
                'ref'), forwarded to IMCFLOW_NOISE_TABLE_FORMAT.
            noise_granularity: Optional sampling granularity ('auto',
                'weight_bitplane', 'input_bitplane'), forwarded to
                IMCFLOW_NOISE_GRANULARITY.
            noise_seed: Optional stochastic noise seed, forwarded to
                IMCFLOW_NOISE_SEED.
            sample_idx: Optional dataset sample index, forwarded as the 9th
                positional arg → gem5 --sample-idx → host binary argv[6].
                When set, host binary writes inputs/outputs under
                sample_<N>/ sub-dirs (per-sample isolation).

        Raises:
            subprocess.CalledProcessError: If simulation fails
        """
        print(f"\n--- Running {self.display_name} ---")

        # Create runner-specific output directory
        output_dir = self.get_output_path(test_name=eval_dir, sample_idx=sample_idx)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        if self.name == "py_runner":
            self._dump_model_input(eval_dir, output_dir, sample_idx)

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

        # Get SRAM_BACKDOOR setting from environment (default: 1 for performance)
        sram_backdoor = os.getenv("SRAM_BACKDOOR", "1")

        # Prepare environment variables for subprocess
        env = os.environ.copy()
        env["SRAM_BACKDOOR"] = sram_backdoor

        # run.sh positional args: binary, gdb, test_name, log_dir, imc_size,
        # noise_csv, noise_layout_json, noise_mode, sample_idx,
        # noise_table_format, noise_granularity, noise_seed. Empty strings mean
        # "skip"; run.sh omits the corresponding --noise-* flag.
        sample_idx_str = "" if sample_idx is None else str(int(sample_idx))
        noise_seed_str = "" if noise_seed is None else str(int(noise_seed))
        sim_command = ["direnv", "exec", ".", "./run.sh",
                       binary_name, gdb_mode, test_name, abs_runner_log_dir, imc_size,
                       noise_csv or "", noise_layout_json or "", noise_mode or "",
                       sample_idx_str, noise_table_format or "",
                       noise_granularity or "", noise_seed_str]

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
    def get_output_path(self, test_name: str, sample_idx: Optional[int] = None) -> str:
        """Get the path to the output file generated by this runner

        Args:
            test_name: Name of the test
            sample_idx: optional dataset sample index; subclasses that thread
                per-sample dump dirs (e.g. py_runner) append ``sample_<N>/``
                to the path.

        Returns:
            Absolute path to output.npy file
        """
        pass

    def cleanup(self) -> None:
        """Perform any cleanup after running (optional, default is no-op)"""
        pass

    def _dump_model_input(self, eval_dir: str, output_path: str,
                          sample_idx: Optional[int]) -> None:
        """Copy the model input consumed by py_runner into its dump directory."""
        if os.path.isabs(eval_dir):
            abs_eval_dir = eval_dir
        else:
            codegen_dir = "/root/project/tvm/tvm_practice/test_imcflow/codegen"
            abs_eval_dir = os.path.abspath(os.path.join(codegen_dir, eval_dir))

        test_inputs_dir = os.path.join(abs_eval_dir, "test_inputs")
        candidates = []
        if sample_idx is not None:
            candidates.append(os.path.join(
                test_inputs_dir, f"sample_{int(sample_idx)}", "model_input.npy"))
        candidates.append(os.path.join(test_inputs_dir, "model_input.npy"))

        src = next((path for path in candidates if os.path.exists(path)), None)
        if src is None:
            print("[warn] model_input.npy not found under test_inputs; "
                  "py_runner input dump skipped")
            return

        dst = os.path.join(os.path.dirname(output_path), "model_input.npy")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Model input dump saved to: {dst}")

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
        # Strip direnv bookkeeping (DIRENV_DIFF / DIRENV_FILE / ...) when invoking
        # 'direnv exec'.  Without this, direnv reverts variables that were exported
        # by an upstream .envrc (e.g. codegen/.envrc's IMCFLOW_DEBUG_COMPUTE) on
        # transition into the target chain — those vars then disappear inside the
        # subprocess.  Stripping makes the target dir's .envrc evaluate from
        # scratch, preserving every other inherited env var.
        if command and command[0] == "direnv":
            env = (dict(env) if env is not None else os.environ.copy())
            for k in [k for k in env if k.startswith("DIRENV_")]:
                env.pop(k, None)

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

    def get_output_path(self, test_name: str, sample_idx: Optional[int] = None) -> str:
        """Get the path to py_runner output file

        Args:
            test_name: Name of the test
            sample_idx: optional dataset sample index; when set, path is
                ``test_outputs/{runner_name}/sample_<N>/output.npy``.

        Returns:
            Absolute path to output.npy in py_runner test_outputs
        """
        # With runner_name support, outputs are saved to:
        # {codegen_dir}/{test_name}/test_outputs/{runner_name}/output.npy
        # test_name is already "eval_dir/xxx_evl" format
        codegen_dir = "/root/project/tvm/tvm_practice/test_imcflow/codegen"
        parts = [codegen_dir, test_name, "test_outputs", self.name]
        if sample_idx is not None:
            parts.append(f"sample_{int(sample_idx)}")
        parts.append("output.npy")
        return os.path.abspath(os.path.join(*parts))


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
        # Both BUGFIX modes use one runner/build directory.  The build manifest
        # notices the effective defines (including a mode change) and rebuilds.
        override = os.environ.get("IMCFLOW_RTL_RUNNER_DIR")
        if override:
            return override
        imcflow_dir = os.environ.get("IMCFLOW_DIR", "/root/project/imcflow")
        return os.path.join(
            imcflow_dir, "pmap", "ISA_sim", "gem5", "tests", "imcflow", "rtl_runner"
        )

    @property
    def timeout(self) -> int:
        return 7200  # 2 hours

    @property
    def display_name(self) -> str:
        return "gem5+RTL Simulation (may take hours)"

    @property
    def main_log_filename(self) -> str:
        return "gem5_rtl.log"

    def run(self, binary_name: str, gdb_mode: str, test_name: str, eval_dir: str,
            noise_csv: Optional[str] = None,
            noise_layout_json: Optional[str] = None,
            noise_mode: Optional[str] = None,
            noise_table_format: Optional[str] = None,
            noise_granularity: Optional[str] = None,
            noise_seed: Optional[int] = None,
            sample_idx: Optional[int] = None) -> None:
        """Run the RTL simulation with automatic port allocation.

        Allocates a unique socket port for VCS-gem5 communication to enable
        concurrent simulations.

        ``noise_csv`` / ``noise_layout_json`` / ``noise_mode`` are accepted for
        interface parity with PyRunner but ignored — the RTL path forwards
        MMIO to VCS, not to the Python IMCU model.

        ``sample_idx`` is accepted for interface parity; the RTL runner's
        bespoke run.sh path doesn't currently route a per-sample suffix to the
        host binary, so it is also ignored here (RTL flow assumes a single
        sample per invocation).
        """
        if noise_csv:
            print(f"[RTLRunner] Ignoring --noise-csv={noise_csv} (RTL path "
                  f"does not exercise the Python IMCU noise model)")
        if noise_layout_json:
            print(f"[RTLRunner] Ignoring --noise-layout-json={noise_layout_json} "
                  f"(RTL path does not exercise the Python IMCU noise model)")
        if noise_mode:
            print(f"[RTLRunner] Ignoring --noise-mode={noise_mode} (RTL path "
                  f"does not exercise the Python IMCU noise model)")
        if noise_table_format:
            print(f"[RTLRunner] Ignoring --noise-table-format={noise_table_format} "
                  f"(RTL path does not exercise the Python IMCU noise model)")
        if noise_granularity:
            print(f"[RTLRunner] Ignoring --noise-granularity={noise_granularity} "
                  f"(RTL path does not exercise the Python IMCU noise model)")
        if noise_seed is not None:
            print(f"[RTLRunner] Ignoring --noise-seed={noise_seed} (RTL path "
                  f"does not exercise the Python IMCU noise model)")
        if sample_idx is not None:
            print(f"[RTLRunner] Ignoring sample_idx={sample_idx} (RTL path "
                  f"does not thread per-sample dump dirs)")
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

            # Build command with socket port as 6th argument
            sim_command = [
                "direnv", "exec", ".", "./run.sh",
                binary_name, gdb_mode, test_name, abs_runner_log_dir, imc_size,
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
        """Ensure that the VCS binary matches the complete build manifest."""
        print(f"Using {self.name} (VCS RTL co-simulation)")

        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"rtl_runner directory not found at: {self.directory_path}")

        from tvm.contrib.imcflow import get_imcflow_bugfix_mode
        bugfix_mode = get_imcflow_bugfix_mode()
        print(f"Checking RTL build manifest (IMCFLOW_BUGFIX={bugfix_mode})")
        compile_command = [
            "direnv", "exec", ".", "make", "ensure-compiled",
            f"IMCFLOW_BUGFIX={bugfix_mode}",
        ]
        compile_log = os.path.join(self.directory_path, "vcs_compile.log")

        try:
            self._stream_command_output(
                command=compile_command,
                cwd=self.directory_path,
                log_path=compile_log,
                timeout=600  # 10 minutes for compilation
            )
            print("VCS RTL build is ready")
        except subprocess.CalledProcessError as e:
            print(f"VCS RTL build setup failed (exit code {e.returncode})")
            print(f"   Check log at: {compile_log}")
            raise RuntimeError(f"VCS RTL build setup failed. See {compile_log} for details") from e
        except subprocess.TimeoutExpired:
            print("VCS RTL build setup timed out after 600 seconds")
            raise RuntimeError("VCS RTL build setup timed out")

    def get_output_path(self, test_name: str, sample_idx: Optional[int] = None) -> str:
        """Get the path to rtl_runner output file

        Args:
            test_name: Name of the test
            sample_idx: kept for signature parity with PyRunner; ignored (the
                RTL path does not thread per-sample dump dirs).

        Returns:
            Absolute path to output.npy in rtl_runner test_outputs
        """
        # With runner_name support, outputs are saved to:
        # {codegen_dir}/{test_name}/test_outputs/{runner_name}/output.npy
        # test_name is already "eval_dir/xxx_evl" format
        codegen_dir = "/root/project/tvm/tvm_practice/test_imcflow/codegen"
        return os.path.abspath(
            os.path.join(codegen_dir, test_name, "test_outputs", self.name, "output.npy")
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
