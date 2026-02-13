"""
Pipeline Options for IMCFlow Test Pipeline

This module provides centralized runtime option management for the IMCFlow
test pipeline, replacing scattered boolean parameters with a clean dataclass-based
approach.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Set, Optional


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
        skip_stages: Set of stages to skip
        rebuild_cpp_only: If True, only rebuild C++ files (skip TVM transforms)
        input_pattern: Input pattern for test data generation
    """
    stop_at: PipelineStage = PipelineStage.COMPARISON
    skip_stages: Set[PipelineStage] = field(default_factory=set)
    rebuild_cpp_only: bool = False
    input_pattern: str = "default"

    def __post_init__(self):
        """Validate options after initialization."""
        self._validate()

    def _validate(self):
        """Centralized option validation.

        Raises:
            ValueError: If options are incompatible
        """
        # skip_setup (TRANSFORM in skip_stages) + rebuild_cpp_only conflict
        if self.should_skip_transform() and self.rebuild_cpp_only:
            raise ValueError(
                "Cannot use --skip-setup and --rebuild-cpp-only together. "
                "--skip-setup reuses all previous outputs, "
                "--rebuild-cpp-only reruns codegen with modified C++ files."
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
        """Check if transform stage should be skipped (skip_setup mode)."""
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

        Runs: transform -> codegen (with rebuild_cpp_only)
        Skips: graph_executor, cpu_validation, simulation, comparison

        Note: This implies rebuild_cpp_only=True for observing memlayout.
        """
        return cls(
            stop_at=PipelineStage.CODEGEN,
            rebuild_cpp_only=True,
            input_pattern=input_pattern
        )

    @classmethod
    def reuse_compiled(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to reuse previously compiled model (skip_setup).

        Skips: transform, codegen, graph_executor
        Runs: cpu_validation -> simulation -> comparison

        Useful for testing different inputs on an already-compiled model.
        """
        return cls(
            stop_at=PipelineStage.COMPARISON,
            skip_stages={
                PipelineStage.TRANSFORM,
                PipelineStage.CODEGEN,
                PipelineStage.GRAPH_EXECUTOR
            },
            input_pattern=input_pattern
        )

    @classmethod
    def rebuild_cpp(cls, input_pattern: str = "default") -> "PipelineOptions":
        """Create options to rebuild C++ files only.

        Skips: transform (uses saved DevConfig state)
        Runs: codegen -> graph_executor -> cpu_validation -> simulation -> comparison

        Used when C++ files in handcraft/ have been modified.
        """
        return cls(
            stop_at=PipelineStage.COMPARISON,
            rebuild_cpp_only=True,
            input_pattern=input_pattern
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        skip_names = [s.name for s in sorted(self.skip_stages)]
        return (
            f"PipelineOptions(stop_at={self.stop_at.name}, "
            f"skip={skip_names}, rebuild_cpp_only={self.rebuild_cpp_only}, "
            f"pattern={self.input_pattern})"
        )


def parse_stop_at(value: str) -> PipelineStage:
    """Parse --stop-at CLI argument to PipelineStage.

    Args:
        value: CLI string value ("transform", "codegen", "compile", "full")

    Returns:
        Corresponding PipelineStage

    Raises:
        ValueError: If value is not recognized
    """
    mapping = {
        "transform": PipelineStage.TRANSFORM,
        "codegen": PipelineStage.CODEGEN,
        "compile": PipelineStage.GRAPH_EXECUTOR,
        "full": PipelineStage.COMPARISON,
    }
    if value.lower() not in mapping:
        valid = ", ".join(mapping.keys())
        raise ValueError(f"Invalid --stop-at value '{value}'. Valid options: {valid}")
    return mapping[value.lower()]
