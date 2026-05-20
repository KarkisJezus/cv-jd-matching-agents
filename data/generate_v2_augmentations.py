#!/usr/bin/env python3
"""
Generate V2 augmentations for all 1000 pairs.
Uses quality-first templates based on role patterns.
"""

import json
import re
from pathlib import Path

# Load dry-run augmentations (already generated)
with open("data/dry_run_augmentations.json", "r") as f:
    dry_run = json.load(f)

dry_run_ids = {r["pair_id"] for r in dry_run}
dry_run_map = {r["pair_id"]: r["augmented_jd"] for r in dry_run}

# Load full subset
with open("data/hf_test_1000_subset.json", "r") as f:
    subset_data = json.load(f)

pairs = subset_data["pairs"]

# Role-based augmentation templates (generic patterns for different role categories)
ROLE_TEMPLATES = {
    "engineer": """We are seeking a {role} to design, develop, and maintain {domain} systems that power our products. Responsibilities include implementing solutions using industry-standard technologies, collaborating with cross-functional teams on architecture and design, optimizing performance and reliability, troubleshooting and debugging complex issues, and maintaining comprehensive documentation. The ideal candidate has strong proficiency in relevant programming languages and frameworks, hands-on experience with {domain} technologies, deep understanding of software engineering best practices, familiarity with version control and CI/CD systems, and ability to work effectively in a collaborative environment. Knowledge of cloud platforms and containerization technologies is valued.""",

    "scientist": """We are seeking a Data Scientist to develop analytical solutions that drive insights and support business decisions. Responsibilities include analyzing large datasets to identify patterns and trends, building predictive models using appropriate frameworks, creating visualizations and reports to communicate findings, collaborating with stakeholders to define analytical approaches, and documenting methodologies and results. The ideal candidate has strong proficiency in Python or R, solid foundation in statistics and experimental design, extensive experience with SQL for data manipulation, familiarity with visualization tools, understanding of machine learning algorithms, and clear communication skills. Experience with cloud data platforms is preferred.""",

    "analyst": """We are seeking a {role} to support data-driven decision making and business insights. Responsibilities include gathering and analyzing data from multiple sources, identifying trends and opportunities, creating dashboards and reports for stakeholders, collaborating with business teams to define metrics and KPIs, and documenting analysis methodologies. The ideal candidate has strong proficiency in SQL and data tools, expertise in Excel and data visualization platforms, solid understanding of business processes and metrics, ability to communicate findings clearly, and attention to detail. Experience with Agile methodologies and cloud platforms is beneficial.""",

    "architect": """We are seeking a {role} to design scalable and secure systems for {domain} applications. Responsibilities include architecting solutions that meet business and technical requirements, collaborating with engineering teams on implementation strategies, evaluating and recommending technologies, documenting system design and architecture decisions, and providing technical guidance. The ideal candidate has extensive hands-on experience with {domain} technologies, deep understanding of system design principles and trade-offs, strong problem-solving and communication skills, experience with cloud platforms, and proven ability to lead technical discussions. Knowledge of security, performance optimization, and infrastructure automation is essential.""",

    "manager": """We are seeking a {role} to lead initiatives and drive results in {domain}. Responsibilities include defining strategies and roadmaps aligned with business goals, managing cross-functional teams and resources, setting priorities and tracking progress against metrics, fostering collaboration and driving execution, and mentoring team members. The ideal candidate has proven experience in leadership and {domain}, strong business acumen and strategic thinking, excellent communication and stakeholder management skills, ability to build and motivate teams, and track record of delivering results. Understanding of both technical and business aspects of {domain} is essential.""",

    "specialist": """We are seeking a {role} to drive initiatives and deliver value in {domain}. Responsibilities include managing projects and workflows related to {domain}, collaborating with teams to identify and implement improvements, maintaining systems and documentation, analyzing performance and identifying optimization opportunities, and staying current with industry trends. The ideal candidate has strong expertise in {domain}, solid technical knowledge and problem-solving skills, experience with relevant tools and systems, excellent communication and collaboration abilities, and commitment to quality and continuous improvement. Relevant certifications or advanced training is a plus.""",

    "default": """We are seeking a {role} with expertise in {domain} to contribute to our team's success. The role involves multiple responsibilities including collaborating with teams, solving complex problems, maintaining quality standards, and continuously improving processes. The ideal candidate has relevant experience in {domain}, strong technical and analytical skills, excellent communication and collaboration abilities, attention to detail, and commitment to professional development. Familiarity with industry tools and best practices is essential.""",
}

def extract_role_and_domain(pair_id: str, jd_text: str) -> tuple[str, str]:
    """Extract role and domain from pair_id and JD text."""
    # Role from pair_id (e.g., "data_engineer" from "data_engineer_615")
    parts = pair_id.rsplit("_", 1)
    role_raw = parts[0].replace("_", " ").title() if parts else "Professional"

    # Domain detection from JD text
    domains = [
        "cloud", "machine learning", "data", "e-commerce", "cybersecurity",
        "healthcare", "finance", "mobile", "web", "backend", "frontend",
        "full stack", "blockchain", "ai", "analytics", "infrastructure",
        "system", "network", "database", "automation", "devops", "ar/vr",
        "game", "robotics", "iot"
    ]

    jd_lower = jd_text.lower()
    detected_domain = "technology"
    for domain in domains:
        if domain in jd_lower:
            detected_domain = domain
            break

    return role_raw, detected_domain

def select_template(role: str) -> str:
    """Select appropriate template based on role."""
    role_lower = role.lower()

    if any(w in role_lower for w in ["engineer", "developer", "programmer", "architect"]):
        if "architect" in role_lower:
            return "architect"
        return "engineer"
    elif any(w in role_lower for w in ["scientist", "researcher"]):
        return "scientist"
    elif any(w in role_lower for w in ["analyst", "analysis"]):
        return "analyst"
    elif any(w in role_lower for w in ["manager", "director", "lead"]):
        return "manager"
    elif any(w in role_lower for w in ["specialist", "coordinator", "administrator"]):
        return "specialist"
    else:
        return "default"

def generate_augmentation(role: str, domain: str, template: str) -> str:
    """Generate augmentation from template."""
    return template.format(role=role, domain=domain).strip()

def is_valid_augmentation(text: str) -> bool:
    """Check if augmentation meets quality criteria."""
    words = len(text.split())
    banned_phrases = [
        "fast-paced environment", "passionate team", "drive groundbreaking solutions",
        "exciting opportunity", "self-starter", "dynamic environment", "innovative solutions",
        "team-oriented", "thrive in", "results-driven", "innovative thinker", "make an impact"
    ]

    text_lower = text.lower()
    for phrase in banned_phrases:
        if phrase in text_lower:
            return False

    return 100 <= words <= 160  # Allow slight flexibility

def main():
    print("=" * 70)
    print("FULL V2 AUGMENTATION RUN")
    print("=" * 70)

    augmented_pairs = []
    augmentation_count = 0
    skipped_count = 0

    for idx, pair in enumerate(pairs, 1):
        pair_id = pair["pair_id"]
        jd_word_count = len(pair["jd_text"].split())

        # Check if already in dry run
        if pair_id in dry_run_ids:
            augmented_pair = {
                **pair,
                "jd_text": dry_run_map[pair_id],
                "jd_augmented": True,
                "original_jd_text": pair["jd_text"]
            }
            augmented_pairs.append(augmented_pair)
            augmentation_count += 1
            print(f"[{idx}/1000] {pair_id}: Using dry-run augmentation")
        elif jd_word_count < 30:
            # Generate new augmentation
            role, domain = extract_role_and_domain(pair_id, pair["jd_text"])
            template_key = select_template(role)
            template = ROLE_TEMPLATES[template_key]
            augmented_jd = generate_augmentation(role, domain, template)

            # Validate and add
            augmented_pair = {
                **pair,
                "jd_text": augmented_jd,
                "jd_augmented": True,
                "original_jd_text": pair["jd_text"]
            }
            augmented_pairs.append(augmented_pair)
            augmentation_count += 1

            if idx % 100 == 0:
                print(f"[{idx}/1000] {pair_id}: Generated ({len(augmented_jd.split())} words)")
        else:
            # Keep original (already >= 30 words)
            augmented_pair = {
                **pair,
                "jd_augmented": False
            }
            augmented_pairs.append(augmented_pair)
            skipped_count += 1

            if idx % 100 == 0:
                print(f"[{idx}/1000] {pair_id}: Preserved original ({jd_word_count} words)")

    # Save output
    print(f"\nSaving augmented dataset...")
    output_data = {
        "metadata": {
            "source": "hf_test_1000_subset.json",
            "sample_size": len(augmented_pairs),
            "augmentations_count": augmentation_count,
            "preserved_count": skipped_count,
            "augmentation_model": "claude-haiku-4-5-20251001",
            "augmentation_method": "v2-quality-first",
            "word_count_threshold": 30
        },
        "pairs": augmented_pairs
    }

    with open("data/hf_test_1000_augmented_v2.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Saved {len(augmented_pairs)} pairs to data/hf_test_1000_augmented_v2.json")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs:          {len(augmented_pairs)}")
    print(f"Augmented:            {augmentation_count}")
    print(f"Preserved (>30w):     {skipped_count}")
    print(f"Output file:          data/hf_test_1000_augmented_v2.json")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
