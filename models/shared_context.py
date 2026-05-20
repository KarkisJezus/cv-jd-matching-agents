"""
SharedContext: the central blackboard for the multi-agent system.

This is the single shared state object that all agents read from and write to.
It implements the Blackboard architectural pattern — agents do not communicate
directly with each other, but instead collaborate through this shared state.

Design decisions:
- Pydantic model for validation and serialization
- Optional fields for scenario-dependent data (enrichment, reflection, memory)
- Built-in logging for evaluation and reproducibility
- Serializable to JSON for experiment storage
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from models.entities import (
    CalibrationOutput,
    CandidateProfile,
    ExtractedEntities,
    FinalDecision,
    IdealCandidateProfile,
    LabeledMemoryEntry,
    LogEntry,
    MemoryEntry,
    NormalizedEntities,
    ReasoningOutput,
    ReflectionOutput,
    SimilarityScores,
)


class SharedContext(BaseModel):
    """
    The shared blackboard state for the agent-based matching system.

    All agents receive this object and can read any field. Each agent
    writes to its designated fields. The orchestrator passes this same
    object through the entire agent chain.

    This differs from a pipeline because:
    - Every agent sees the FULL state, not just its predecessor's output
    - Agents can adapt their behavior based on any field
    - The reflection loop can reset fields and trigger re-execution
    """

    # ── Input (set once at creation, never modified) ──────────────

    cv_text: str = Field(description="Original CV text")
    jd_text: str = Field(description="Original job description text")
    scenario: Literal["A", "B", "C"] = Field(
        default="A",
        description="Which scenario to run: A=basic, B=+enrichment, C=+memory",
    )

    # ── Extraction (written by ExtractionAgent) ───────────────────

    cv_entities: Optional[ExtractedEntities] = Field(
        default=None,
        description="Entities extracted from the CV",
    )
    jd_entities: Optional[ExtractedEntities] = Field(
        default=None,
        description="Entities extracted from the job description",
    )

    # ── Enrichment (written by ContextEnrichmentAgent, Scenario B+) ─

    normalized_entities: Optional[NormalizedEntities] = Field(
        default=None,
        description="Normalized and taxonomy-mapped entities",
    )
    enrichment_notes: list[str] = Field(
        default_factory=list,
        description="Notes from the enrichment process",
    )

    # ── Matching (written by SemanticMatchingAgent) ────────────────

    similarity_scores: Optional[SimilarityScores] = Field(
        default=None,
        description="Semantic similarity results",
    )

    # ── Reasoning (written by ReasoningAgent) ─────────────────────

    reasoning_output: Optional[ReasoningOutput] = Field(
        default=None,
        description="LLM-generated reasoning about match quality",
    )

    # ── Reflection (written by ReflectionAgent, Phase 3+) ─────────

    reflection_output: Optional[ReflectionOutput] = Field(
        default=None,
        description="Reflection agent's review of the reasoning",
    )
    needs_revision: bool = Field(
        default=False,
        description="Flag set by ReflectionAgent to trigger re-reasoning",
    )
    revision_count: int = Field(
        default=0,
        description="How many revision cycles have occurred",
    )
    max_revisions: int = Field(
        default=2,
        description="Maximum allowed revision cycles",
    )

    # ── Decision (written by DecisionAgent) ───────────────────────

    final_decision: Optional[FinalDecision] = Field(
        default=None,
        description="Final matching decision",
    )

    # ── Tier 2 — Profiles (replace cv_entities/jd_entities + normalized_entities)
    # Produced by CVProfilingAgent and JDProfilingAgent. Both old and new fields
    # coexist during the migration; downstream agents read whichever is populated.

    cv_profile: Optional[CandidateProfile] = Field(
        default=None,
        description="Candidate profile from CVProfilingAgent (replaces cv_entities + half of normalized_entities)",
    )
    jd_profile: Optional[IdealCandidateProfile] = Field(
        default=None,
        description="Ideal-candidate profile from JDProfilingAgent, with ESCO role context",
    )

    # ── Tier 2 — Two-pass decision (Scenario C) ──────────────────
    # In the new flow, DecisionAgent commits initial_decision (Pass 1), then
    # CalibrationAgent reviews labeled memory and produces final_decision (Pass 2).
    # In Scenarios A/B, only final_decision is populated.

    initial_decision: Optional[FinalDecision] = Field(
        default=None,
        description="Pass 1 decision committed without seeing labeled memory (Scenario C only)",
    )
    calibration_output: Optional[CalibrationOutput] = Field(
        default=None,
        description="Output of CalibrationAgent comparing initial_decision to labeled history",
    )

    # ── Memory (Scenario C) ───────────────────────────────────────

    memory_entries: list[MemoryEntry] = Field(
        default_factory=list,
        description="LEGACY: Retrieved unlabeled memories (old MemoryEntry format)",
    )
    labeled_memory_entries: list[LabeledMemoryEntry] = Field(
        default_factory=list,
        description="Tier 2: Retrieved labeled past pairs (decision + ground-truth label)",
    )

    # ── Logging (appended by all agents) ──────────────────────────

    logs: list[LogEntry] = Field(
        default_factory=list,
        description="Chronological log of all agent actions",
    )

    # Per-agent token usage accumulated across this run.
    # Maps agent_name -> {prompt_tokens, completion_tokens, calls}.
    # Populated automatically by BaseAgent.execute() for agents that use an LLM.
    agent_token_usage: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-agent token usage for context-size analysis",
    )

    # ── Utility methods ──────────────────────────────────────────

    def add_log(self, agent_name: str, action: str, details: str = "", duration: float = 0.0):
        """Convenience method to add a log entry."""
        self.logs.append(
            LogEntry(
                agent_name=agent_name,
                action=action,
                details=details,
                duration_seconds=duration,
            )
        )

    def has_enrichment(self) -> bool:
        """Check if enrichment data is available (for agent adaptation)."""
        return self.normalized_entities is not None

    def has_reflection(self) -> bool:
        """Check if reflection has been performed."""
        return self.reflection_output is not None

    def has_memory(self) -> bool:
        """Check if (legacy) memory entries are available."""
        return len(self.memory_entries) > 0

    def has_labeled_memory(self) -> bool:
        """Check if Tier 2 labeled memory entries are available."""
        return len(self.labeled_memory_entries) > 0

    def has_profiles(self) -> bool:
        """Check if Tier 2 profiles are populated (replaces has-extraction check for new flow)."""
        return self.cv_profile is not None and self.jd_profile is not None

    def get_skills_for_matching(self) -> tuple[list[str], list[str]]:
        """
        Return the best available skill lists for matching.

        Priority (Tier 2 first, then Tier 1 fallbacks):
          1. cv_profile + jd_profile (Tier 2)
          2. normalized_entities (Tier 1 with enrichment)
          3. cv_entities + jd_entities (Tier 1 raw)

        For Tier 2, the JD-side combines required_skills + typical_role_skills
        (deduped) so the matcher has the full skill set the role typically demands.
        """
        # Tier 2: profiles (richest signal, preferred when present)
        if self.cv_profile and self.jd_profile:
            cv_skills = list(self.cv_profile.skills)
            # JD: union of explicit requirements + ESCO-typical role skills, deduped
            seen = set()
            jd_skills: list[str] = []
            for s in (
                list(self.jd_profile.required_skills)
                + list(self.jd_profile.typical_role_skills)
            ):
                k = s.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    jd_skills.append(s)
            return cv_skills, jd_skills

        # Tier 1 with enrichment
        if self.has_enrichment() and self.normalized_entities:
            cv_skills = [s.normalized for s in self.normalized_entities.cv_skills]
            jd_skills = [s.normalized for s in self.normalized_entities.jd_skills]
            return cv_skills, jd_skills

        # Tier 1 raw extraction
        if self.cv_entities and self.jd_entities:
            return list(self.cv_entities.skills), list(self.jd_entities.skills)

        return [], []
