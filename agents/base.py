"""
BaseAgent: abstract base class for all agents in the system.

Every agent in the system inherits from this class. It enforces:
1. A consistent interface (execute method)
2. Automatic logging of agent execution
3. Time tracking for performance evaluation

The base class is intentionally simple — the goal is to provide
a common interface without imposing unnecessary abstractions.
"""

import time
from abc import ABC, abstractmethod

from models.shared_context import SharedContext


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Each agent receives the full SharedContext (blackboard) and can
    read any field. Agents write to their designated fields.

    The execute() method is the main entry point. It wraps the
    agent's process() method with logging and timing.
    """

    @property
    def name(self) -> str:
        """Agent name, defaults to class name."""
        return self.__class__.__name__

    def execute(self, context: SharedContext) -> SharedContext:
        """
        Run this agent on the shared context.

        This method:
        1. Logs the start of execution
        2. Snapshots LLM token usage before/after (if agent uses LLM)
        3. Calls the agent's process() method
        4. Logs completion with timing and per-agent token cost
        5. Returns the modified context

        Subclasses should NOT override this method.
        Override process() instead.
        """
        context.add_log(self.name, "started")
        start_time = time.time()

        # Snapshot LLM usage before this agent runs (if applicable)
        before_usage = self._snapshot_llm_usage()

        try:
            context = self.process(context)
            duration = time.time() - start_time

            # Compute per-agent token delta
            after_usage = self._snapshot_llm_usage()
            token_details = ""
            if before_usage is not None and after_usage is not None:
                delta_prompt = after_usage["prompt_tokens"] - before_usage["prompt_tokens"]
                delta_completion = after_usage["completion_tokens"] - before_usage["completion_tokens"]
                delta_total = delta_prompt + delta_completion
                delta_calls = after_usage["total_calls"] - before_usage["total_calls"]
                if delta_calls > 0:
                    context.agent_token_usage[self.name] = context.agent_token_usage.get(
                        self.name, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                    )
                    context.agent_token_usage[self.name]["prompt_tokens"] += delta_prompt
                    context.agent_token_usage[self.name]["completion_tokens"] += delta_completion
                    context.agent_token_usage[self.name]["calls"] += delta_calls
                    token_details = (
                        f", tokens={delta_total} "
                        f"(prompt={delta_prompt}, completion={delta_completion}, "
                        f"calls={delta_calls})"
                    )

                    # Context Engineering 2.0: warn if any single prompt approaches
                    # the 50% context-window fullness threshold (Hua et al., 2025)
                    if delta_prompt > 64000:  # ~50% of 128K context window
                        context.add_log(
                            self.name,
                            "context_size_warning",
                            f"Prompt size {delta_prompt} tokens exceeds ~50% of typical "
                            f"128K context window — quality may degrade.",
                        )

            context.add_log(
                self.name,
                "completed",
                details=f"Finished in {duration:.2f}s{token_details}",
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            context.add_log(
                self.name,
                "failed",
                details=f"Error after {duration:.2f}s: {str(e)}",
                duration=duration,
            )
            raise

        return context

    def _snapshot_llm_usage(self) -> dict | None:
        """
        Snapshot the current LLM usage counters if this agent uses an LLM.

        Agents that use an LLM store the client on `self._llm`. Returns
        None for agents without an LLM (MemoryRetrievalAgent, SemanticMatchingAgent).
        """
        llm = getattr(self, "_llm", None)
        if llm is None:
            return None
        usage = getattr(llm, "usage", None)
        if usage is None or not isinstance(usage, dict):
            return None
        return dict(usage)  # defensive copy

    @abstractmethod
    def process(self, context: SharedContext) -> SharedContext:
        """
        Agent-specific processing logic. Must be implemented by subclasses.

        This method receives the full shared context and should:
        1. Read whatever fields it needs from context
        2. Perform its specific task
        3. Write its results to the appropriate context fields
        4. Return the modified context

        Args:
            context: The shared blackboard state

        Returns:
            The modified shared context
        """
        ...
