"""
JudgeAgent: an independent LLM-as-Judge auditor for CV-JD matching.

This is a meta-evaluator: it reads CV + JD + the source dataset's claimed label,
and produces an independent verdict on whether the source label is correct.

Design choices follow Gu et al. 2025, "A Survey on LLM-as-a-Judge"
(arXiv:2411.15594):

1. **Cross-family judge** — the JudgeAgent uses a different LLM provider than
   the system being evaluated, to avoid the self-enhancement bias documented in
   §4.2.1 of the survey. The default provider is DeepSeek (configurable via
   JUDGE_BASE_URL / JUDGE_MODEL); switching to Gemini is three env-var edits.

2. **Anonymous evaluation** — the judge never sees the system's predictions
   (Tier 1 score, Tier 2 score, calibration decisions, etc.). This avoids
   compassion-fade bias (§4.2.2) where the judge anchors to system output.

3. **Chain-of-Thought decomposition** (§3.1.1.a) — the prompt structures the
   reasoning as numbered steps the judge must work through in order, rather
   than asking for a verdict in one shot.

4. **Criteria decomposition** (§3.1.1.b) — instead of a single "is this a match?"
   judgment, the judge produces sub-scores per dimension (skills coverage,
   experience match, seniority match, education match, domain alignment).
   These are aggregated into the final verdict.

5. **Few-shot examples in the system prompt** (§3.1.1) — three worked examples:
   one clear match, one clear reject, one genuinely ambiguous. The third
   teaches the judge that "ambiguous" is a valid verdict.

6. **Structured JSON output** (§3.1.2) — robust to LLM stochasticity and
   straightforward to post-process.

7. **Anti-bias instructions** (§4.2.2) — explicit prompt-level guards against
   length bias and concreteness bias that LLM judges are known to exhibit.

8. **Confidence-gated "ambiguous" verdict** — when the judge's confidence
   falls below a threshold (default 0.7), the source assessment defaults to
   "ambiguous" rather than forcing a binary call. This explicitly carves out
   the noise floor of the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.client import BaseLLMClient


AMBIGUOUS_CONFIDENCE_THRESHOLD = 0.7


SYSTEM_PROMPT = """\
You are an independent expert reviewer of CV-job description matches. Your task
is to assess whether a candidate's CV represents a reasonable fit for a given
job description, and to evaluate whether the dataset's claimed match/no-match
label is correct.

You will see:
- A candidate's CV
- A job description
- A "source label" claiming whether the candidate was selected (true) or
  rejected (false)

You will NOT see any system predictions or scores. Base your judgment purely on
the candidate's qualifications versus the JD's stated requirements.

== Method ==

Work through these steps in order, then output your verdict:

Step 1 — JD analysis: List the JD's explicit requirements (required skills,
experience years, seniority level, education, domain). Only what the JD
literally states; do not infer.

Step 2 — CV analysis: List the candidate's relevant qualifications.

Step 3 — Coverage check: For each requirement from Step 1, is it MET, PARTIAL,
or MISSING in the CV? Cite specific evidence from the CV when present.

Step 4 — Severity: Of the MISSING items, which are load-bearing for this role?
A senior role missing senior experience is severe; a peripheral nice-to-have
is not.

Step 5 — Verdict: Decide match / no-match / ambiguous, with confidence 0-1.
"Ambiguous" is a valid verdict — use it when reasonable hiring managers would
reasonably disagree, or when the JD/CV is too thin to support a confident call.

Step 6 — Source assessment: Compare your verdict to the source label.
- "correct" — the source label matches your verdict, OR your verdict is
  ambiguous AND the source label is plausible
- "incorrect" — your verdict clearly disagrees with the source
- "ambiguous" — even after analysis you cannot confidently judge whether the
  source label is correct

Step 7 — Failure mode: Classify the dominant signal of this pair into ONE
category. Use this label even when source_assessment is "correct" — it
characterizes the pair, not the error.
- "good_match"        — clear match: CV meets JD requirements
- "categorical_mismatch" — completely different field/role
- "experience_gap"    — right role but insufficient years/seniority
- "skill_mismatch"    — same domain but missing specific required skills
- "seniority_mismatch" — over- or under-qualified
- "domain_mismatch"   — adjacent domain but not the JD's
- "templated_reject"  — JD trivially asks for the candidate's exact role yet
                        source labeled it "rejected" (dataset noise pattern)
- "ambiguous_jd"      — JD too thin or contradictory to evaluate against
- "other"             — none of the above

== Anti-bias instructions ==

- Do NOT favor longer documents. A concise CV that meets requirements is as
  strong as a verbose one.
- Do NOT over-trust specific numbers (e.g. "increased revenue by 31.7%").
  Numerical specificity indicates detail, not necessarily quality.
- Do NOT infer information not present in the CV or JD. If a JD doesn't
  mention cloud experience, do not penalize a CV for lacking it.
- Do NOT assume the source label is correct. About 20-40% of source labels
  in this dataset are templated and unreliable. Your job is to assess.

== Output format ==

Return ONLY a JSON object with these exact fields. No prose outside the JSON.

{
  "step_1_jd_requirements": ["<list of explicit JD requirements>"],
  "step_2_cv_qualifications": ["<list of relevant CV qualifications>"],
  "step_3_coverage": [
    {"requirement": "<from step 1>", "status": "MET|PARTIAL|MISSING",
     "evidence": "<specific CV/JD reference>"}
  ],
  "step_4_severity": "<one sentence: which gaps are load-bearing>",
  "step_5_verdict": "match|no-match|ambiguous",
  "step_5_confidence": <float 0-1>,
  "step_6_source_assessment": "correct|incorrect|ambiguous",
  "step_7_failure_mode": "good_match|categorical_mismatch|experience_gap|skill_mismatch|seniority_mismatch|domain_mismatch|templated_reject|ambiguous_jd|other",
  "rationale": "<2-3 sentences citing specific CV/JD content, NOT step numbers>",
  "criterion_scores": {
    "skills_coverage": <float 0-1>,
    "experience_match": <float 0-1>,
    "seniority_match": <float 0-1>,
    "education_match": <float 0-1>,
    "domain_alignment": <float 0-1>
  }
}

== Worked examples ==

EXAMPLE 1 — Clear match.
JD: "Looking for a senior Data Engineer with 5+ years building ETL pipelines."
CV: 7 years as Data Engineer, leads ETL pipeline projects on AWS.
Source label: true (selected).

Verdict: {
  "step_5_verdict": "match", "step_5_confidence": 0.92,
  "step_6_source_assessment": "correct",
  "step_7_failure_mode": "good_match",
  "rationale": "JD requires senior Data Engineer with 5+ years ETL experience. \
CV shows 7 years as Data Engineer leading ETL pipeline projects. Direct fit.",
  "criterion_scores": {"skills_coverage": 0.9, "experience_match": 1.0,
    "seniority_match": 1.0, "education_match": 0.8, "domain_alignment": 1.0}
}

EXAMPLE 2 — Clear reject.
JD: "Senior cloud architect, 8+ years on AWS or Azure."
CV: Junior frontend developer, 1 year React experience.
Source label: false (rejected).

Verdict: {
  "step_5_verdict": "no-match", "step_5_confidence": 0.95,
  "step_6_source_assessment": "correct",
  "step_7_failure_mode": "categorical_mismatch",
  "rationale": "JD requires senior cloud architect with 8+ years on AWS/Azure. \
CV is a junior frontend developer with no cloud architecture experience. \
Categorical mismatch on role and seniority.",
  "criterion_scores": {"skills_coverage": 0.05, "experience_match": 0.1,
    "seniority_match": 0.1, "education_match": 0.5, "domain_alignment": 0.05}
}

EXAMPLE 3 — Genuinely ambiguous.
JD: "Looking for an experienced Data Analyst to drive solutions in AI research."
CV: 5 years Data Analyst (Excel, SQL, Tableau, Python). No ML/AI work shown.
Source label: false (rejected).

Verdict: {
  "step_5_verdict": "ambiguous", "step_5_confidence": 0.55,
  "step_6_source_assessment": "ambiguous",
  "step_7_failure_mode": "ambiguous_jd",
  "rationale": "JD asks for experienced Data Analyst (clearly met by CV) but \
adds 'AI research' context. CV shows no ML/AI work. Reasonable hiring managers \
could reject (no AI experience) or accept (data analyst role with on-the-job \
ramp-up). The source rejection is plausible but not the only defensible call.",
  "criterion_scores": {"skills_coverage": 0.7, "experience_match": 0.85,
    "seniority_match": 0.7, "education_match": 0.6, "domain_alignment": 0.5}
}
"""


VALID_FAILURE_MODES = {
    "good_match", "categorical_mismatch", "experience_gap", "skill_mismatch",
    "seniority_mismatch", "domain_mismatch", "templated_reject",
    "ambiguous_jd", "other",
}


def get_prompt_version_hash() -> str:
    """SHA256 hash of the SYSTEM_PROMPT, truncated to 12 hex chars.

    Recorded in audit metadata so you can prove which prompt revision produced
    a given results file. If the prompt changes, the hash changes; reviewers
    can verify reproducibility.
    """
    import hashlib
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


@dataclass
class JudgeResult:
    """Structured output from one judge call on one pair."""
    pair_id: str
    judge_label: Optional[bool]      # True/False; None if ambiguous
    judge_confidence: float           # 0-1
    source_assessment: str            # "correct" | "incorrect" | "ambiguous"
    rationale: str
    criterion_scores: dict[str, float] = field(default_factory=dict)
    step_1_jd_requirements: list[str] = field(default_factory=list)
    step_2_cv_qualifications: list[str] = field(default_factory=list)
    step_3_coverage: list[dict] = field(default_factory=list)
    step_4_severity: str = ""
    failure_mode: str = "other"       # taxonomy label; see VALID_FAILURE_MODES
    raw_verdict: str = ""             # "match" | "no-match" | "ambiguous"
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None       # set if the judge call failed


class JudgeAgent:
    """Independent LLM-as-Judge auditor for CV-JD pairs.

    The agent is stateless — each call to `judge()` is independent. Token
    counters live on the underlying LLM client, so to track per-judge usage
    cleanly the caller should give each judge call its own client OR snapshot
    usage before/after.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def judge(
        self,
        pair_id: str,
        cv_text: str,
        jd_text: str,
        source_label: Optional[bool],
    ) -> JudgeResult:
        """Run the judge on one pair. Returns a structured JudgeResult.

        On parsing failure, returns a JudgeResult with `error` set. The caller
        decides whether to retry, skip, or surface.
        """
        source_label_str = (
            "true (candidate was selected)" if source_label is True
            else "false (candidate was rejected)" if source_label is False
            else "unknown"
        )
        user_prompt = (
            f"== Candidate CV ==\n{cv_text}\n\n"
            f"== Job Description ==\n{jd_text}\n\n"
            f"== Source label ==\n{source_label_str}\n\n"
            f"Now produce the JSON verdict per the format and method above."
        )

        # Snapshot usage so per-judge token counts are correct even if the
        # client is shared across calls.
        before = self._snapshot_usage()
        try:
            raw = self._llm.chat_json(SYSTEM_PROMPT, user_prompt)
        except (json.JSONDecodeError, ValueError) as e:
            after = self._snapshot_usage()
            return JudgeResult(
                pair_id=pair_id, judge_label=None, judge_confidence=0.0,
                source_assessment="ambiguous",
                rationale=f"Judge call failed: {e}",
                tokens_in=after["prompt_tokens"] - before["prompt_tokens"],
                tokens_out=after["completion_tokens"] - before["completion_tokens"],
                error=str(e),
            )
        after = self._snapshot_usage()

        return self._parse_judgment(
            pair_id=pair_id,
            raw=raw,
            tokens_in=after["prompt_tokens"] - before["prompt_tokens"],
            tokens_out=after["completion_tokens"] - before["completion_tokens"],
        )

    def _snapshot_usage(self) -> dict:
        usage = getattr(self._llm, "usage", None)
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    def _parse_judgment(
        self, pair_id: str, raw: dict, tokens_in: int, tokens_out: int,
    ) -> JudgeResult:
        verdict = str(raw.get("step_5_verdict", "ambiguous")).lower().strip()
        if verdict not in ("match", "no-match", "ambiguous"):
            verdict = "ambiguous"

        try:
            confidence = float(raw.get("step_5_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # If confidence is below the threshold, override the verdict to ambiguous
        # AND the source assessment to ambiguous. This implements the survey's
        # recommendation to carve out a low-confidence band rather than forcing
        # a binary call.
        if confidence < AMBIGUOUS_CONFIDENCE_THRESHOLD:
            verdict = "ambiguous"

        if verdict == "match":
            judge_label: Optional[bool] = True
        elif verdict == "no-match":
            judge_label = False
        else:
            judge_label = None  # ambiguous

        source_assessment = str(raw.get("step_6_source_assessment", "ambiguous")).lower().strip()
        if source_assessment not in ("correct", "incorrect", "ambiguous"):
            source_assessment = "ambiguous"
        if confidence < AMBIGUOUS_CONFIDENCE_THRESHOLD:
            source_assessment = "ambiguous"

        # Sanitize criterion_scores to floats in [0, 1]
        cs_raw = raw.get("criterion_scores", {}) or {}
        criterion_scores = {}
        for k, v in cs_raw.items():
            try:
                criterion_scores[str(k)] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue

        def _str_list(field_name: str) -> list[str]:
            v = raw.get(field_name, [])
            if not isinstance(v, list):
                return []
            return [str(x) for x in v]

        coverage = raw.get("step_3_coverage", [])
        if not isinstance(coverage, list):
            coverage = []
        # Filter to dict items only
        coverage = [c for c in coverage if isinstance(c, dict)]

        # Failure-mode taxonomy: snap to a known label, else "other"
        failure_mode = str(raw.get("step_7_failure_mode", "other")).lower().strip()
        if failure_mode not in VALID_FAILURE_MODES:
            failure_mode = "other"

        return JudgeResult(
            pair_id=pair_id,
            judge_label=judge_label,
            judge_confidence=confidence,
            source_assessment=source_assessment,
            rationale=str(raw.get("rationale", "")).strip(),
            criterion_scores=criterion_scores,
            step_1_jd_requirements=_str_list("step_1_jd_requirements"),
            step_2_cv_qualifications=_str_list("step_2_cv_qualifications"),
            step_3_coverage=coverage,
            step_4_severity=str(raw.get("step_4_severity", "")).strip(),
            failure_mode=failure_mode,
            raw_verdict=verdict,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
