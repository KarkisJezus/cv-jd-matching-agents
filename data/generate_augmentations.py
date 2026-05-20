#!/usr/bin/env python3
"""
Generate augmented JDs directly based on role and domain inference.
This approach produces realistic 2-3 paragraph job descriptions
that preserve the role/domain combination from the original short JD.
"""

import json
import re
from pathlib import Path

# Template-based augmentation that preserves role and domain while expanding
AUGMENTATION_TEMPLATES = {
    "default": """We are seeking a talented {role} to join our {domain} team. As a key member of our organization, you will be responsible for designing and implementing solutions that drive innovation and business value. Your technical expertise and ability to collaborate with cross-functional teams will be essential to our success.

In this role, you will have the opportunity to work on challenging projects that span {domain} technologies and {role} responsibilities. You should have strong problem-solving skills, the ability to communicate complex ideas clearly, and a passion for continuous learning. We value team members who take ownership of their work and are committed to delivering high-quality results.

If you are a {role} with experience in {domain} and are ready to make an impact, we encourage you to apply. Join us in building the next generation of innovative solutions.""",

    "developer": """We are looking for an experienced {role} to lead our {domain} initiatives. In this role, you'll architect and implement solutions using cutting-edge technologies while mentoring junior team members. Your experience with {domain} technologies will be crucial as we expand our platform.

You'll collaborate with product managers, designers, and other engineers to translate complex requirements into elegant technical solutions. Your strong foundation in software engineering principles, combined with domain expertise in {domain}, will enable you to drive technical excellence across our teams.

The ideal candidate brings several years of professional experience as a {role}, with demonstrable expertise in {domain} development. If you're excited about solving complex technical challenges in {domain}, we'd love to hear from you.""",

    "analyst": """Join our analytics team as a {role} and help us unlock insights from our {domain} data. You'll be responsible for collecting, analyzing, and presenting data-driven recommendations that inform strategic business decisions. Your expertise in {domain} analytics will directly impact how we measure success and optimize our operations.

Working closely with stakeholders across the organization, you'll develop dashboards, reports, and visualizations that tell compelling stories about our {domain} performance. You should be comfortable with large datasets, statistical analysis, and communicating findings to both technical and non-technical audiences.

We're looking for a detail-oriented {role} with strong analytical skills and experience in the {domain} space. Your ability to translate raw data into actionable insights will be essential to our continued growth.""",

    "manager": """We are recruiting a strategic {role} to lead initiatives in the {domain} space. You will oversee cross-functional teams, set strategic direction, and drive execution of key projects. Your leadership experience and domain knowledge in {domain} will be instrumental in achieving our ambitious goals.

As a {role}, you'll be responsible for defining priorities, allocating resources, and ensuring our teams deliver exceptional results. You'll mentor team members, foster a collaborative culture, and work with executive leadership to shape our {domain} strategy. Your strong communication and stakeholder management skills will be critical.

The successful candidate will bring significant experience as a {role}, with a proven track record in the {domain} industry. If you're ready to lead and make a strategic impact, we invite you to apply.""",

    "specialist": """We are seeking a skilled {role} to join our {domain} practice. You'll bring specialized expertise that strengthens our capabilities and allows us to deliver better solutions for our clients and partners. Your knowledge of {domain} best practices and emerging trends will be invaluable.

In this position, you'll work on diverse projects, solving complex {domain} challenges while continuously expanding your technical knowledge. You'll have opportunities to mentor colleagues, contribute to process improvements, and shape the evolution of our {domain} offerings. Your attention to detail and commitment to excellence will set you apart.

If you are a dedicated {role} with deep expertise in {domain} and a passion for delivering quality work, this could be the perfect opportunity for you to grow your career.""",
}

def extract_role_and_domain(jd_text: str) -> tuple[str, str]:
    """Extract the primary role and any domain mention from short JD."""
    # List of known roles (simplified)
    roles = [
        "Software Engineer", "Developer", "Data Scientist", "Data Analyst", "Data Engineer",
        "Product Manager", "Project Manager", "Business Analyst", "QA Engineer",
        "DevOps Engineer", "Cloud Engineer", "Cloud Architect", "ML Engineer",
        "Machine Learning Engineer", "AI Engineer", "AI Researcher", "Analyst",
        "Administrator", "System Administrator", "Database Administrator", "Network Engineer",
        "Architect", "Manager", "Specialist", "Designer", "UI Designer", "UX Designer",
        "AR/VR Developer", "Blockchain Developer", "Game Developer", "Mobile Developer",
        "Full Stack Developer", "Backend Engineer", "Frontend Engineer", "Content Writer",
        "Cybersecurity Analyst", "HR Specialist", "Robotics Engineer", "Graphic Designer",
        "Digital Marketing Specialist", "E-commerce Specialist", "IT Support Specialist"
    ]

    # List of common domains (simplified)
    domains = [
        "AI", "machine learning", "ML", "data science", "cloud", "AWS", "Azure", "healthcare",
        "fintech", "finance", "e-commerce", "retail", "education", "blockchain", "crypto",
        "cybersecurity", "security", "DevOps", "infrastructure", "performance", "scalability",
        "mobile", "web", "game development", "AR/VR", "augmented reality", "virtual reality",
        "IoT", "embedded systems", "robotics", "automation", "analytics", "big data",
        "real-time", "streaming", "databases", "SQL", "NoSQL", "microservices", "API"
    ]

    jd_lower = jd_text.lower()

    # Extract role
    detected_role = "Professional"
    for role in roles:
        if role.lower() in jd_lower:
            detected_role = role
            break

    # Extract domain
    detected_domain = "technology"
    for domain in domains:
        if domain.lower() in jd_lower:
            detected_domain = domain
            break

    return detected_role, detected_domain

def select_template(jd_text: str, role: str) -> str:
    """Select appropriate template based on role."""
    role_lower = role.lower()

    if any(w in role_lower for w in ["developer", "engineer", "architect"]):
        return "developer"
    elif any(w in role_lower for w in ["analyst", "science"]):
        return "analyst"
    elif any(w in role_lower for w in ["manager", "director", "lead"]):
        return "manager"
    elif any(w in role_lower for w in ["specialist", "admin"]):
        return "specialist"
    else:
        return "default"

def augment_jd(jd_text: str) -> str:
    """Augment a short JD using template-based expansion."""
    role, domain = extract_role_and_domain(jd_text)
    template_key = select_template(jd_text, role)
    template = AUGMENTATION_TEMPLATES[template_key]

    # Generate augmented text
    augmented = template.format(role=role, domain=domain)
    return augmented

def main():
    print("=" * 70)
    print("AUGMENTATION: Template-based JD Expansion")
    print("=" * 70)

    # Load sampled pairs
    print("\nLoading sampled pairs...")
    with open("data/sampled_1000_pairs.json", "r") as f:
        data = json.load(f)

    pairs = data["pairs"]
    print(f"[OK] Loaded {len(pairs)} pairs")

    # Process all pairs
    print("\nAugmenting JDs...")
    augmented_pairs = []
    augmentation_count = 0

    for i, pair in enumerate(pairs):
        jd_word_count = len(pair["jd_text"].split())

        if jd_word_count < 30:
            augmented_jd = augment_jd(pair["jd_text"])
            augmented_pair = {
                **pair,
                "jd_text": augmented_jd,
                "jd_augmented": True,
                "original_jd_text": pair["jd_text"]
            }
            augmentation_count += 1
        else:
            augmented_pair = {
                **pair,
                "jd_augmented": False
            }

        augmented_pairs.append(augmented_pair)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}] {augmentation_count} augmented so far...")

    # Save output
    print(f"\nSaving augmented dataset...")
    output_data = {
        "metadata": {
            "source": "hf_test_rest.json",
            "sample_size": len(augmented_pairs),
            "augmentations_count": augmentation_count,
            "word_count_threshold": 30,
            "augmentation_method": "template-based"
        },
        "pairs": augmented_pairs
    }

    with open("data/hf_test_1000_augmented.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Saved {len(augmented_pairs)} pairs to data/hf_test_1000_augmented.json")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs:       {len(augmented_pairs)}")
    print(f"Augmented:         {augmentation_count}")
    print(f"Preserved (>30w):  {len(augmented_pairs) - augmentation_count}")
    print(f"Output file:       data/hf_test_1000_augmented.json")

    # Show 3 examples
    print(f"\n{'='*70}")
    print("3 BEFORE/AFTER EXAMPLES")
    print(f"{'='*70}\n")

    short_pairs = [p for p in augmented_pairs if p.get("jd_augmented")]

    for i, pair in enumerate(short_pairs[:3], 1):
        print(f"Example {i}: {pair['pair_id']}")
        orig = pair["original_jd_text"]
        aug = pair["jd_text"]
        print(f"\nORIGINAL ({len(orig.split())} words):")
        print(f'"{orig}"')
        print(f"\nAUGMENTED ({len(aug.split())} words):")
        print(f'"{aug}"')
        print()

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
