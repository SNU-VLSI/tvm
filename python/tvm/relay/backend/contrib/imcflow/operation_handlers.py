"""Handler registry and base classes for IMCFlow operation code generation.

This module provides a pluggable architecture for generating code blocks from
relay operations. New operations can be supported by creating handler classes
and registering them with the @register_operation_handler decorator.

Handlers receive a BuilderContext that wraps the relay.Call with helper methods:
    def handle(self, call_ctx: BuilderContext) -> None:
        hid = call_ctx.get_hid()
        in_edges = call_ctx.get_input_edges()
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from tvm import relay

if TYPE_CHECKING:
    from .codegen import ImceCodeBlockBuilder
    from .builder_context import BuilderContext


class OperationHandler(ABC):
    """Base class for operation-specific code generators.

    Each handler encapsulates the logic for generating IMCFlow code blocks
    from a specific type of relay operation.

    Handlers receive a BuilderContext object that wraps the call and provides
    all necessary helper methods and shared state. The builder reference is
    stored in self.builder for special cases that need traversal.
    """

    def __init__(self):
        """Initialize handler with no builder reference yet."""
        self.builder: 'Optional[ImceCodeBlockBuilder]' = None

    def set_builder(self, builder: 'ImceCodeBlockBuilder') -> None:
        """Set the builder reference for this handler.

        Called by the registry when a handler is about to be used.

        Args:
            builder: The ImceCodeBlockBuilder instance
        """
        self.builder = builder

    @abstractmethod
    def can_handle(self, call: relay.Call) -> bool:
        """Check if this handler can process the given call.

        Args:
            call: The relay Call expression to check

        Returns:
            True if this handler should process the call, False otherwise
        """
        pass

    @abstractmethod
    def handle(self, call_ctx: 'BuilderContext') -> None:
        """Generate code blocks for the operation.

        Args:
            call_ctx: BuilderContext wrapping the call with helper methods

        Note:
            Access self.builder if you need to call builder.visit() or
            check builder state (rare, mainly for CompositeHandler).
        """
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Handler priority (lower number = higher priority).

        Priority determines the order in which handlers are checked.
        Composite handlers should have priority 0 to be checked first.
        Regular operation handlers typically use priority 10.

        Returns:
            Integer priority value
        """
        pass


class OperationHandlerRegistry:
    """Registry for operation handlers with priority-based dispatch.

    Handlers are registered via the @register_operation_handler decorator
    and are checked in priority order when dispatching operations.

    This registry automatically wraps calls in BuilderContext before passing
    to handlers.
    """

    def __init__(self):
        self._handlers: list[OperationHandler] = []

    def register(self, handler: OperationHandler) -> None:
        """Register a handler and sort by priority.

        Args:
            handler: The OperationHandler instance to register
        """
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: h.priority)

    def get_handler(self, call: relay.Call, builder: 'ImceCodeBlockBuilder') -> Optional[OperationHandler]:
        """Find the first matching handler for a call.

        Args:
            call: The relay Call expression to find a handler for
            builder: The ImceCodeBlockBuilder instance

        Returns:
            The first matching OperationHandler, or None if no handler matches
        """
        for handler in self._handlers:
            # Set builder reference before checking
            handler.set_builder(builder)
            if handler.can_handle(call):
                return handler
        return None

    def handle(self, call: relay.Call, builder: 'ImceCodeBlockBuilder') -> bool:
        """Dispatch call to appropriate handler with automatic wrapping.

        This method automatically wraps the call in a BuilderContext before
        passing it to the handler. The handler's builder reference is set
        before calling can_handle.

        Args:
            call: The relay Call expression to dispatch
            builder: The ImceCodeBlockBuilder instance

        Returns:
            True if a handler processed the call, False otherwise
        """
        handler = self.get_handler(call, builder)
        if handler:
            # Import here to avoid circular dependency
            from .builder_context import BuilderContext

            # Wrap the call with builder's shared state
            call_ctx = BuilderContext(
                call=call,
                edges=builder.edges,
                codeblocks=builder.codeblocks,
                curr_composite_id=builder.curr_composite_id,
                curr_conv_block=builder.curr_conv_block,
                last_tuple_idx=builder.last_tuple_idx
            )

            # Call handler with wrapped context (builder already set)
            handler.handle(call_ctx)

            # Update builder state in case handler modified it
            builder.curr_composite_id = call_ctx.curr_composite_id
            builder.curr_conv_block = call_ctx.curr_conv_block
            builder.last_tuple_idx = call_ctx.last_tuple_idx

            return True
        return False


# Global registry instance
_HANDLER_REGISTRY = OperationHandlerRegistry()


def register_operation_handler(handler_class: type) -> type:
    """Decorator to register an operation handler.

    Usage:
        @register_operation_handler
        class MyOpHandler(OperationHandler):
            def priority(self) -> int:
                return 10

            def can_handle(self, call, builder) -> bool:
                return call.op == op.get("my_op")

            def handle(self, call, builder) -> None:
                # Generate code blocks
                pass

    Args:
        handler_class: The OperationHandler class to register

    Returns:
        The same handler class (allows normal use as a class)
    """
    _HANDLER_REGISTRY.register(handler_class())
    return handler_class


def get_handler_registry() -> OperationHandlerRegistry:
    """Get the global handler registry.

    Returns:
        The global OperationHandlerRegistry instance
    """
    return _HANDLER_REGISTRY
