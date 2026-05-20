"""
Structured data models for the agent-based matching system.

These models define the data that agents read from and write to
the shared context (blackboard). Each model represents a specific
type of information produced during the matching process.
"""

import uuid
from datetime import datetime

from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Extraction models
# ──────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    """Entities extracted from a CV or job description by the ExtractionAgent."""

    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    experience: list[str] = Field(default_factory=list, description="Work experience entries")
    education: list[str] = Field(default_factory=list, description="Education qualifications")
    languages: list[str] = Field(default_factory=list, description="Spoken/written languages")
    certifications: list[str] = Field(default_factory=list, description="Certifications")
    raw_summary: str = Field(default="", description="LLM-generated brief summary of the document")


# ──────────────────────────────────────────────
# Normalization / enrichment models (Scenario B+)
# ──────────────────────────────────────────────

class NormalizedSkill(BaseModel):
    """A single skill after normalization and optional taxonomy mapping."""

    original: str = Field(description="Original skill text as extracted")
    normalized: str = Field(description="Cleaned / canonical form")
    esco_code: Optional[str] = Field(default=None, description="ESCO taxonomy code if mapped")
    esco_label: Optional[str] = Field(default=None, description="ESCO taxonomy label if mapped")
    synonyms: list[str] = Field(default_factory=list, description="Known synonyms found")


class NormalizedEntities(BaseModel):
    """Normalized entity sets for both CV and JD."""

    cv_skills: list[NormalizedSkill] = Field(default_factory=list)
    jd_skills: list[NormalizedSkill] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Matching models
# ──────────────────────────────────────────────

class SkillMatch(BaseModel):
    """A single skill-to-skill match with similarity score."""

    cv_skill: str = Field(description="Skill from the CV")
    jd_skill: str = Field(description="Skill from the job description")
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")
    match_type: str = Field(
        default="semantic",
        description="How the match was determined: 'exact', 'semantic', 'taxonomy'",
    )


class SimilarityScores(BaseModel):
    """Aggregated similarity results from the SemanticMatchingAgent."""

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall semantic similarity")
    skill_matches: list[SkillMatch] = Field(default_factory=list, description="Individual matches")
    matched_skills_count: int = Field(default=0, description="How many JD skills were matched")
    total_jd_skills: int = Field(default=0, description="Total JD skills to match against")
    coverage_ratio: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Fraction of JD skills covered by CV (matched/total)",
    )


# ──────────────────────────────────────────────
# Reasoning models
# ──────────────────────────────────────────────

class ReasoningOutput(BaseModel):
    """Structured output from the ReasoningAgent's LLM analysis."""

    strengths: list[str] = Field(default_factory=list, description="Candidate strengths")
    gaps: list[str] = Field(default_factory=list, description="Missing skills or experience")
    concerns: list[str] = Field(default_factory=list, description="Potential concerns")
    overall_assessment: str = Field(default="", description="Free-text overall assessment")
    suggested_score: float = Field(
        ge=0.0, le=100.0, default=0.0,
        description="Agent's suggested match score before reflection",
    )


# ──────────────────────────────────────────────
# Reflection models (Phase 3+)
# ──────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    """Output from the ReflectionAgent's review of the reasoning."""

    is_consistent: bool = Field(
        default=True,
        description="Whether reasoning is consistent with the data",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific inconsistencies or gaps in reasoning",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving the analysis",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Reflection agent's confidence in the reasoning quality",
    )
    revision_reason: str = Field(
        default="",
        description="Why revision was requested (if needs_revision is True)",
    )


# ──────────────────────────────────────────────
# Decision models
# ──────────────────────────────────────────────

class FinalDecision(BaseModel):
    """The final matching decision produced by the DecisionAgent."""

    score: float = Field(ge=0.0, le=100.0, description="Final match score 0-100")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    recommendation: str = Field(
        default="",
        description="One of: 'strong_match', 'good_match', 'partial_match', 'weak_match', 'no_match'",
    )
    explanation: str = Field(default="", description="Human-readable explanation")
    key_factors: list[str] = Field(
        default_factory=list,
        description="Main factors that influenced the decision",
    )


# ──────────────────────────────────────────────
# Memory models (Scenario C)
# ──────────────────────────────────────────────

class MemoryEntry(BaseModel):
    """A stored memory from a previous matching run, used in Scenario C."""

    memory_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Stable unique identifier for this memory",
    )
    cv_summary: str = Field(default="", description="Summary of the CV that was matched")
    jd_summary: str = Field(default="", description="Summary of the JD that was matched")
    decision_score: float = Field(ge=0.0, le=100.0, description="Score from that run")
    reasoning_summary: str = Field(default="", description="Key reasoning points")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When this memory was created",
    )
    similarity_to_current: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="How similar this memory is to the current query",
    )
    # Context Engineering 2.0 Selection Factor 2: Logical Dependency.
    # Tracks which past memories were retrieved and seen by the reasoning
    # agent when this decision was made. Enables dependency-chain traversal
    # for interpretability and memory impact analysis (Hua et al., 2025 §6.3).
    influenced_by: list[str] = Field(
        default_factory=list,
        description="memory_ids of past decisions that influenced this one",
    )


# ──────────────────────────────────────────────
# Tier 2 (new architecture) — Profiling models
# ──────────────────────────────────────────────
# These extend the original ExtractedEntities concept with interpretive fields
# (seniority, archetype, role classification). They are produced by the new
# CVProfilingAgent and JDProfilingAgent which replace the single ExtractionAgent.
# The originals (ExtractedEntities, NormalizedEntities) are kept for backward
# compatibility while the new pipeline is being built and tested.

class CandidateProfile(BaseModel):
    """
    A profile of WHO is applying — produced by CVProfilingAgent from a CV.

    Extends raw entity extraction with interpretive fields that describe
    "what kind of person this is," not just a list of skills. The richer
    representation feeds the matching and reasoning agents.
    """

    # Raw entity fields (carried over from ExtractedEntities for compat)
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    experience: list[str] = Field(default_factory=list, description="Work experience entries")
    education: list[str] = Field(default_factory=list, description="Education qualifications")
    languages: list[str] = Field(default_factory=list, description="Spoken/written languages")
    certifications: list[str] = Field(default_factory=list, description="Certifications")

    # Interpretive fields — the new value-add of profiling
    seniority_level: str = Field(
        default="unknown",
        description="One of: junior, mid, senior, lead, unknown",
    )
    domain_expertise: list[str] = Field(
        default_factory=list,
        description="Domain areas the candidate appears strong in (e.g., 'machine learning', 'cloud infrastructure')",
    )
    candidate_archetype: str = Field(
        default="",
        description="One-sentence summary of who this candidate is",
    )
    likely_role_fit: str = Field(
        default="",
        description="ESCO occupation the candidate's profile most resembles, if any",
    )
    raw_summary: str = Field(default="", description="2-3 sentence freeform summary")


class IdealCandidateProfile(BaseModel):
    """
    A profile of the IDEAL CANDIDATE the JD is seeking — produced by JDProfilingAgent.

    Combines the JD's explicit requirements with ESCO role context (typical
    skills/experience for the role) so downstream reasoning has a richer
    target to match against than just the JD's literal text.
    """

    # Raw extracted requirements from JD text
    required_skills: list[str] = Field(default_factory=list, description="Skills the JD explicitly requires")
    typical_role_skills: list[str] = Field(
        default_factory=list,
        description="Skills typically expected for this role per ESCO, even if not in JD text",
    )
    required_experience_years: float = Field(
        default=0.0,
        description="Years of experience the JD requires (0 if not specified)",
    )
    required_education: str = Field(default="", description="Education requirements from the JD")

    # Role classification (the ESCO bridge)
    detected_role: str = Field(
        default="",
        description="The ESCO role this JD describes (e.g., 'machine_learning_engineer')",
    )
    role_confidence: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="How confident the agent is in the role classification",
    )
    esco_code: str = Field(default="", description="ESCO occupation code if matched")

    # Interpretive fields
    seniority_required: str = Field(
        default="unknown",
        description="One of: junior, mid, senior, lead, unknown",
    )
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="Main responsibilities the role entails",
    )
    raw_summary: str = Field(default="", description="2-3 sentence freeform summary")


# ──────────────────────────────────────────────
# Tier 2 — Calibration model (Pass 2 output)
# ──────────────────────────────────────────────

class CalibrationOutput(BaseModel):
    """
    Output of CalibrationAgent (Pass 2 of the two-pass decision flow).

    Compares the initial_decision (Pass 1) against retrieved labeled memories
    to detect systematic miscalibration and proposes a final adjusted score.
    """

    calibration_decision: str = Field(
        default="keep",
        description="One of: 'lower', 'raise', 'keep'",
    )
    adjusted_score: float = Field(
        ge=0.0, le=100.0,
        description="The calibrated final score (may equal initial_decision.score if 'keep')",
    )
    adjusted_recommendation: str = Field(
        default="",
        description="Recommendation derived from adjusted_score (strong_match/good_match/...)",
    )
    rationale: str = Field(
        default="",
        description="Explanation citing the labeled memories that justified the adjustment",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Confidence in the calibration decision",
    )
    pattern_observed: str = Field(
        default="",
        description="The pattern detected in past similar pairs (e.g., 'system over-scores by 10pt for this profile')",
    )
    n_supporting_memories: int = Field(
        default=0,
        description="How many retrieved memories supported the calibration decision",
    )


# ──────────────────────────────────────────────
# Tier 2 — Labeled memory model
# ──────────────────────────────────────────────

class LabeledMemoryEntry(BaseModel):
    """
    A memory entry containing both the system's past decision AND the human ground-truth label.

    Used in the new Scenario C streaming-feedback architecture. After each pair is evaluated
    (system decision committed), the human label from the dataset is attached and the entry
    is saved. Subsequent pairs retrieve these entries to calibrate against past mistakes.

    Methodologically critical: the system_score is committed BEFORE the human label is
    attached. The label leaks into memory only AFTER the prediction is locked in. See the
    StreamBench protocol (Yehudai et al. 2025 §3) for the formal framing.
    """

    memory_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Stable unique identifier",
    )
    pair_id: str = Field(
        default="",
        description="The dataset's pair_id this memory came from. Used for overlap detection.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )

    # Profiles (replacing the old free-text summaries)
    cv_profile_summary: str = Field(default="", description="Summary from CandidateProfile")
    jd_profile_summary: str = Field(default="", description="Summary from IdealCandidateProfile")
    detected_role: str = Field(default="", description="Role classification at the time of decision")

    # System's decision (committed BEFORE label was attached)
    system_score: float = Field(ge=0.0, le=100.0, description="System's final score (post-calibration)")
    system_recommendation: str = Field(default="", description="System's final recommendation")
    system_initial_score: float = Field(
        ge=0.0, le=100.0, default=0.0,
        description="System's Pass 1 score (before calibration adjustment)",
    )
    system_reasoning_summary: str = Field(
        default="",
        description="Brief summary of why the system decided as it did",
    )

    # Human ground truth (attached AFTER prediction)
    ground_truth_label: bool = Field(
        description="True = match (select), False = no_match (reject)",
    )
    ground_truth_reason: str = Field(
        default="",
        description="The human's reason from the dataset (Reason_for_decision field)",
    )

    # Derived fields (computed once both are known)
    was_correct: bool = Field(
        description="Whether the system's predicted_label matched the ground_truth_label",
    )
    error_direction: str = Field(
        default="",
        description="One of: TP (true positive), FP (false positive), TN, FN",
    )

    # Logical-dependency chain (per Hua et al. 2025 §6.3)
    influenced_by: list[str] = Field(
        default_factory=list,
        description="memory_ids of past entries consulted when making this decision",
    )

    # Set at retrieval time (not at storage time)
    similarity_to_current: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Cosine similarity to the current query — populated when retrieved",
    )


# ──────────────────────────────────────────────
# Logging models
# ──────────────────────────────────────────────

class LogEntry(BaseModel):
    """A single log entry tracking agent activity for evaluation."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )
    agent_name: str = Field(description="Which agent produced this log")
    action: str = Field(description="What the agent did")
    details: str = Field(default="", description="Additional details or data")
    duration_seconds: float = Field(default=0.0, description="How long the action took")
