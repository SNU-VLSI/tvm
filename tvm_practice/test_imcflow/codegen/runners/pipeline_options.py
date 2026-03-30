"""
Pipeline Options for IMCFlow Test Pipeline

This module provides centralized runtime option management for the IMCFlow
test pipeline, replacing scattered boolean parameters with a clean dataclass-based
approach.
"""

from enum import IntEnum, Enum, auto
from dataclasses import dataclass, field
from typing import Set, Optional


class RunMode(Enum):
    """Execution mode for the pipeline.

    Determines how the pipeline handles model compilation and reuse.
    """
    FULL = "full"      # Full pipeline: transform -> codegen -> compile -> simulate
    REUSE = "reuse"    # Reuse compiled model: skip to simulation
    PATCH = "patch"    # Apply handcraft patches: codegen -> patch -> compile -> simulate


class PipelineStage(IntEnum):
    """Pipeline stages in execution order.

    IntEnum allows natural comparison operators (< <= > >=).
    Lower values execute first.
    """
    TRANSFORM = 1       # Frontend transformation (fe_only stops here)
    CODEGEN = 2         # Code generation (codegen_only stops here)
    GRAPH_EXECUTOR = 3  # Graph executor build (compile_only stops here)
    CPU_VALIDATION = 4  # CPU reference validation
    SIMULATION = 5      # Hardware simulation
    COMPARISON = 6      # Output comparison (full run)


@dataclass
class PipelineOptions:
    """Centralized runtime options for the IMCFlow test pipeline.

    This dataclass replaces the scattered boolean parameters (skip_setup,
    rebuild_modified_cpp, codegen_only, compile_only, fe_only) with a
    cleaner, validated configuration object.

    Attributes:
        stop_at: Pipeline stage to stop at (inclusive)
        start_at: Pipeline stage to start at (skips earlier stages)
        skip_stages: Set of stages to skip (computed from start_at)
        with_patch: If True, apply handcraft patches before codegen
        input_pattern: Input pattern for test data generation
        run_mode: Computed execution mode (FULL, REUSE, or PATCH)
    """
    stop_at: PipelineStage = PipelineStage.COMPARISON
    start_at: PipelineStage = PipelineStage.TRANSFORM
    skip_stages: Set[PipelineStage] = field(default_factory=set)
    with_patch: bool = False
    single_qconv: bool = False
    input_pattern: str = "default"
    dataset: Optional[str] = None
    sample: Optional[int] = None
    retry_disable: bool = False
    max_retry_count: Optional[int] = None
    run_mode: RunMode = field(init=False)

    def __post_init__(self):
        """Compute skip_stages and run_mode from options, then validate."""
        # Compute skip_stages based on start_at
        if self.start_at > PipelineStage.TRANSFORM:
            # Skip all stages before start_at
            for stage in PipelineStage:
                if stage < self.start_at:
                    self.skip_stages.add(stage)

        # Compute run_mode based on with_patch and start_at
        if self.with_patch:
            self.run_mode = RunMode.PATCH
        elif self.start_at >= PipelineStage.CPU_VALIDATION:
            self.run_mode = RunMode.REUSE
        else:
            self.run_mode = RunMode.FULL

        self._validate()

    def _validate(self):
        """Centralized option validation.

        Raises:
            ValueError: If options are incompatible
        """
        # start_at must not be after stop_at
        if self.start_at > self.stop_at:
            raise ValueError(
                f"--start-at ({self.start_at.name.lower()}) cannot be after "
                f"--stop-at ({self.stop_at.name.lower()}). "
                f"Nothing would be executed."
            )

        # Can't skip transform if stopping at transform
        if self.stop_at == PipelineStage.TRANSFORM and self.should_skip_transform():
            raise ValueError(
                "Cannot skip transform stage when stopping at transform. "
                "Nothing would be executed."
            )

    def should_run(self, stage: PipelineStage) -> bool:
        """Check if a given stage should run.

        Args:
            stage: The pipeline stage to check

        Returns:
            True if the stage should execute
        """
        return stage <= self.stop_at and stage not in self.skip_stages

    def should_skip_transform(self) -> bool:
        """Check if transform stage should be skipped (start_at > TRANSFORM)."""
        return PipelineStage.TRANSFORM in self.skip_stages

    def should_skip_codegen(self) -> bool:
        """Check if codegen stage should be skipped."""
        return not self.should_run(PipelineStage.CODEGEN)

    def should_skip_graph_executor(self) -> bool:
        """Check if graph executor stage should be skipped."""
        return not self.should_run(PipelineStage.GRAPH_EXECUTOR)

    def should_skip_cpu_validation(self) -> bool:
        """Check if CPU validation stage should be skipped."""
        return not self.should_run(PipelineStage.CPU_VALIDATION)

    def should_skip_simulation(self) -> bool:
        """Check if simulation stage should be skipped."""
        return not self.should_run(PipelineStage.SIMULATION)

    # =========================================================================
    # Factory Methods - Create common configurations
    # =========================================================================

    @classmethod
    def full_run(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options for a full pipeline run.

        Runs all stages: transform -> codegen -> graph_executor ->
        cpu_validation -> simulation -> comparison
        """
        return cls(
            stop_at=PipelineStage.COMPARISON,
            input_pattern=input_pattern
        )

    @classmethod
    def compile_only(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to stop after graph executor compilation.

        Runs: transform -> codegen -> graph_executor
        Skips: cpu_validation, simulation, comparison
        """
        return cls(
            stop_at=PipelineStage.GRAPH_EXECUTOR,
            input_pattern=input_pattern
        )

    @classmethod
    def frontend_only(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to stop after frontend transformation.

        Runs: transform only
        Skips: codegen, graph_executor, cpu_validation, simulation, comparison
        """
        return cls(
            stop_at=PipelineStage.TRANSFORM,
            input_pattern=input_pattern
        )

    @classmethod
    def codegen_only(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to stop after codegen (for C++ debugging).

        Runs: codegen only (skips transform)
        Skips: transform, graph_executor, cpu_validation, simulation, comparison
        """
        return cls(
            stop_at=PipelineStage.CODEGEN,
            start_at=PipelineStage.CODEGEN,
            input_pattern=input_pattern
        )

    @classmethod
    def reuse_compiled(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to reuse previously compiled model.

        Skips: transform, codegen, graph_executor
        Runs: cpu_validation -> simulation -> comparison

        Useful for testing different inputs on an already-compiled model.
        """
        return cls(
            stop_at=PipelineStage.COMPARISON,
            start_at=PipelineStage.SIMULATION,
            input_pattern=input_pattern
        )

    @classmethod
    def with_patch_run(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to run with handcraft patches.

        Runs: codegen -> graph_executor -> cpu_validation -> simulation -> comparison
        Skips: transform (uses saved DevConfig state)

        Used when C++ files in handcraft/ have been modified.
        """
        return cls(
            stop_at=PipelineStage.COMPARISON,
            start_at=PipelineStage.CODEGEN,
            with_patch=True,
            input_pattern=input_pattern
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        skip_names = [s.name for s in sorted(self.skip_stages)]
        dataset_str = f", dataset={self.dataset}[{self.sample}]" if self.dataset else ""
        return (
            f"PipelineOptions(start_at={self.start_at.name}, stop_at={self.stop_at.name}, "
            f"skip={skip_names}, with_patch={self.with_patch}, "
            f"pattern={self.input_pattern}{dataset_str}, run_mode={self.run_mode.name})"
        )


# Unified stage name mapping (verb-based, consistent across --start-at and --stop-at)
STAGE_NAMES = {
    "transform": PipelineStage.TRANSFORM,
    "codegen": PipelineStage.CODEGEN,
    "compile": PipelineStage.GRAPH_EXECUTOR,
    "validate": PipelineStage.CPU_VALIDATION,
    "simulate": PipelineStage.SIMULATION,
    "compare": PipelineStage.COMPARISON,
}


def parse_stop_at(value: str) -> PipelineStage:
    """Parse --stop-at CLI argument to PipelineStage.

    Args:
        value: CLI string value (transform, codegen, compile, validate, simulate, compare)

    Returns:
        Corresponding PipelineStage

    Raises:
        ValueError: If value is not recognized
    """
    # Support legacy "full" as alias for "compare"
    if value.lower() == "full":
        value = "compare"
    # Support legacy "simulation" as alias for "simulate"
    if value.lower() == "simulation":
        value = "simulate"

    if value.lower() not in STAGE_NAMES:
        valid = ", ".join(STAGE_NAMES.keys())
        raise ValueError(f"Invalid --stop-at value '{value}'. Valid options: {valid}")
    return STAGE_NAMES[value.lower()]


def parse_start_at(value: str) -> PipelineStage:
    """Parse --start-at CLI argument to PipelineStage.

    Args:
        value: CLI string value (transform, codegen, compile, validate, simulate)

    Returns:
        Corresponding PipelineStage

    Raises:
        ValueError: If value is not recognized
    """
    # Support legacy "simulation" as alias for "simulate"
    if value.lower() == "simulation":
        value = "simulate"

    # --start-at doesn't allow "compare" (nothing would execute)
    valid_start_stages = {k: v for k, v in STAGE_NAMES.items() if k != "compare"}

    if value.lower() not in valid_start_stages:
        valid = ", ".join(valid_start_stages.keys())
        raise ValueError(f"Invalid --start-at value '{value}'. Valid options: {valid}")
    return valid_start_stages[value.lower()]
