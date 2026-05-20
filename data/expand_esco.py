"""
Expand the ESCO occupations file with 30 additional roles plus hr_specialist.

Adds breadth across tech, product, marketing/sales, HR/operations, healthcare,
and creative domains. Also adds hr_specialist as an explicit shorthand for
human_resources_specialist (some JDs use the abbreviated form).

Run once:
    python data/expand_esco.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
ESCO_PATH = ROOT / "data" / "esco_occupations.json"


NEW_ROLES = {
    # --- Tech variants (12) ---
    "frontend_developer": {
        "esco_code": "S2.1.4", "preferred_label": "frontend developer",
        "alt_labels": ["UI developer", "client-side engineer", "web frontend engineer"],
        "description": "Builds user-facing web interfaces using HTML, CSS, JavaScript and modern frameworks like React, Vue, or Angular.",
        "typical_skills": ["JavaScript/TypeScript", "React or Vue or Angular", "HTML/CSS", "responsive design", "browser APIs", "build tools (Vite/Webpack)"],
        "typical_experience_years": 2,
        "typical_education": "BSc in CS or self-taught with portfolio",
        "typical_responsibilities": ["UI implementation", "cross-browser compatibility", "component design", "accessibility", "performance optimization"],
        "seniority_levels": ["junior", "mid", "senior", "staff"],
    },
    "backend_developer": {
        "esco_code": "S2.1.5", "preferred_label": "backend developer",
        "alt_labels": ["server-side engineer", "API developer", "backend engineer"],
        "description": "Builds server-side application logic, APIs, databases, and services that power applications.",
        "typical_skills": ["Python or Java or Node.js or Go", "REST/GraphQL APIs", "SQL databases", "caching", "message queues", "system design"],
        "typical_experience_years": 2,
        "typical_education": "BSc in CS or equivalent practical experience",
        "typical_responsibilities": ["API design and implementation", "database modelling", "service integration", "performance tuning", "monitoring"],
        "seniority_levels": ["junior", "mid", "senior", "staff", "principal"],
    },
    "ios_developer": {
        "esco_code": "S2.1.6", "preferred_label": "iOS developer",
        "alt_labels": ["iPhone developer", "Apple platform developer", "Swift developer"],
        "description": "Builds native iOS applications using Swift and Apple's SDKs.",
        "typical_skills": ["Swift", "SwiftUI/UIKit", "Xcode", "Core Data", "App Store deployment", "iOS Human Interface Guidelines"],
        "typical_experience_years": 2,
        "typical_education": "BSc in CS or equivalent",
        "typical_responsibilities": ["iOS app development", "App Store submission", "performance profiling on devices", "OS version compatibility"],
        "seniority_levels": ["junior", "mid", "senior", "lead"],
    },
    "android_developer": {
        "esco_code": "S2.1.7", "preferred_label": "Android developer",
        "alt_labels": ["Kotlin developer", "Android engineer"],
        "description": "Builds native Android applications using Kotlin or Java and the Android SDK.",
        "typical_skills": ["Kotlin", "Android SDK", "Jetpack Compose", "Room/SQLite", "Play Store deployment", "Material Design"],
        "typical_experience_years": 2,
        "typical_education": "BSc in CS or equivalent",
        "typical_responsibilities": ["Android app development", "Play Store releases", "device fragmentation handling", "battery and memory optimization"],
        "seniority_levels": ["junior", "mid", "senior", "lead"],
    },
    "embedded_engineer": {
        "esco_code": "S2.2.1", "preferred_label": "embedded systems engineer",
        "alt_labels": ["embedded software engineer", "firmware engineer (loose)"],
        "description": "Develops software that runs directly on hardware — microcontrollers, RTOS systems, IoT devices.",
        "typical_skills": ["C/C++", "RTOS (FreeRTOS, Zephyr)", "low-level debugging", "hardware interfaces (I2C/SPI/UART)", "memory-constrained design"],
        "typical_experience_years": 3,
        "typical_education": "BSc in EE or CS",
        "typical_responsibilities": ["firmware development", "device driver implementation", "hardware-software integration", "low-level performance tuning"],
        "seniority_levels": ["junior", "mid", "senior", "principal"],
    },
    "firmware_engineer": {
        "esco_code": "S2.2.2", "preferred_label": "firmware engineer",
        "alt_labels": ["firmware developer", "low-level software engineer"],
        "description": "Writes firmware for hardware devices — bootloaders, drivers, low-level OS components.",
        "typical_skills": ["C/Assembly", "bootloader development", "memory management", "interrupt handling", "device drivers", "JTAG debugging"],
        "typical_experience_years": 3,
        "typical_education": "BSc in EE or Embedded Systems",
        "typical_responsibilities": ["firmware implementation", "bringing up new boards", "hardware bring-up", "field-update mechanisms"],
        "seniority_levels": ["mid", "senior", "principal"],
    },
    "release_engineer": {
        "esco_code": "S2.3.1", "preferred_label": "release engineer",
        "alt_labels": ["build engineer", "build and release engineer"],
        "description": "Owns software build, packaging, and release pipelines; ensures repeatable, traceable deployments.",
        "typical_skills": ["CI/CD (Jenkins, GitHub Actions, GitLab)", "build systems (Bazel, Make)", "containers (Docker)", "scripting (Bash, Python)", "release automation"],
        "typical_experience_years": 3,
        "typical_education": "BSc in CS or equivalent practical experience",
        "typical_responsibilities": ["maintaining build pipelines", "release coordination", "artifact management", "rollback automation"],
        "seniority_levels": ["mid", "senior", "lead"],
    },
    "integration_engineer": {
        "esco_code": "S2.3.2", "preferred_label": "integration engineer",
        "alt_labels": ["systems integration engineer", "API integration engineer"],
        "description": "Connects disparate systems through APIs, ETL, and middleware so they exchange data reliably.",
        "typical_skills": ["REST/SOAP APIs", "ETL tools", "middleware (MuleSoft, Boomi)", "data mapping", "messaging (Kafka, RabbitMQ)", "scripting"],
        "typical_experience_years": 3,
        "typical_education": "BSc in CS or IT",
        "typical_responsibilities": ["building integration flows", "data transformation", "third-party API consumption", "monitoring integration health"],
        "seniority_levels": ["mid", "senior", "lead"],
    },
    "solutions_architect": {
        "esco_code": "S2.4.1", "preferred_label": "solutions architect",
        "alt_labels": ["technical solutions architect", "enterprise solutions architect"],
        "description": "Designs end-to-end technical solutions to meet business requirements, often across multiple systems.",
        "typical_skills": ["enterprise architecture patterns", "cloud platforms", "system design at scale", "stakeholder communication", "technology evaluation"],
        "typical_experience_years": 7,
        "typical_education": "BSc in CS plus extensive experience; sometimes MBA",
        "typical_responsibilities": ["solution design", "architecture documentation", "presenting to stakeholders", "vendor evaluation", "technical leadership"],
        "seniority_levels": ["senior", "principal"],
    },
    "technical_writer": {
        "esco_code": "S2.5.1", "preferred_label": "technical writer",
        "alt_labels": ["documentation engineer", "API documentation specialist"],
        "description": "Produces user-facing and developer documentation, API references, and technical guides.",
        "typical_skills": ["clear technical writing", "Markdown/AsciiDoc", "documentation tooling (Sphinx, Docusaurus)", "version control", "screenshot/diagram tools"],
        "typical_experience_years": 2,
        "typical_education": "BA/BSc in technical communication, English, or CS",
        "typical_responsibilities": ["writing user guides", "API documentation", "release notes", "documentation site maintenance"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "mlops_engineer": {
        "esco_code": "S3.1.1", "preferred_label": "MLOps engineer",
        "alt_labels": ["ML platform engineer", "machine learning operations engineer"],
        "description": "Builds infrastructure and pipelines for training, deploying, and monitoring ML models at scale.",
        "typical_skills": ["Python", "Kubernetes", "ML pipeline tools (Airflow, Kubeflow, MLflow)", "container orchestration", "model serving", "monitoring/observability"],
        "typical_experience_years": 4,
        "typical_education": "BSc/MSc in CS or related; ML fundamentals",
        "typical_responsibilities": ["ML pipeline automation", "model deployment", "feature store maintenance", "drift detection", "infrastructure"],
        "seniority_levels": ["mid", "senior", "staff"],
    },
    "security_engineer": {
        "esco_code": "S2.6.1", "preferred_label": "security engineer",
        "alt_labels": ["application security engineer", "infrastructure security engineer"],
        "description": "Builds and maintains technical security controls in software and infrastructure.",
        "typical_skills": ["secure coding", "threat modeling", "OWASP Top 10", "SAST/DAST tooling", "network security", "incident response"],
        "typical_experience_years": 4,
        "typical_education": "BSc in CS plus security certifications (CISSP, OSCP)",
        "typical_responsibilities": ["security architecture review", "vulnerability assessment", "incident response", "security tooling integration"],
        "seniority_levels": ["mid", "senior", "principal"],
    },

    # --- Product/UX (3) ---
    "product_designer": {
        "esco_code": "S4.1.1", "preferred_label": "product designer",
        "alt_labels": ["UX/UI product designer", "digital product designer"],
        "description": "Combines UX research, interaction design, and visual design to define product experiences end-to-end.",
        "typical_skills": ["Figma", "user research", "interaction design", "design systems", "prototyping", "usability testing"],
        "typical_experience_years": 3,
        "typical_education": "BFA/BA in design or HCI",
        "typical_responsibilities": ["end-to-end product design", "user research synthesis", "design system contribution", "cross-functional collaboration"],
        "seniority_levels": ["junior", "mid", "senior", "staff"],
    },
    "product_owner": {
        "esco_code": "S4.2.1", "preferred_label": "product owner",
        "alt_labels": ["agile product owner", "scrum product owner"],
        "description": "Represents the customer's voice in agile teams; owns the backlog and prioritizes work.",
        "typical_skills": ["agile/Scrum", "backlog management", "user story writing", "stakeholder communication", "roadmapping"],
        "typical_experience_years": 3,
        "typical_education": "BA/BSc in business, CS, or related",
        "typical_responsibilities": ["backlog grooming", "sprint planning", "acceptance criteria", "stakeholder updates"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "technical_product_manager": {
        "esco_code": "S4.2.2", "preferred_label": "technical product manager",
        "alt_labels": ["TPM", "platform product manager", "infrastructure product manager"],
        "description": "Product manager for developer-facing or infrastructure products; deeply technical.",
        "typical_skills": ["technical depth in target domain", "API design literacy", "roadmapping", "data analysis", "developer empathy"],
        "typical_experience_years": 5,
        "typical_education": "BSc in CS or engineering plus product experience",
        "typical_responsibilities": ["roadmap definition", "spec writing for technical products", "developer adoption strategies", "API/SDK product decisions"],
        "seniority_levels": ["mid", "senior", "principal"],
    },

    # --- Marketing/Sales (5) ---
    "copywriter": {
        "esco_code": "S5.1.1", "preferred_label": "copywriter",
        "alt_labels": ["advertising copywriter", "marketing copywriter"],
        "description": "Writes persuasive marketing and advertising copy across channels.",
        "typical_skills": ["compelling writing", "brand voice", "SEO basics", "A/B testing copy", "content briefs"],
        "typical_experience_years": 2,
        "typical_education": "BA in English, journalism, marketing, or related",
        "typical_responsibilities": ["ad copy", "email campaigns", "landing page copy", "brand voice consistency"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "social_media_manager": {
        "esco_code": "S5.1.2", "preferred_label": "social media manager",
        "alt_labels": ["social media specialist", "community manager"],
        "description": "Owns brand presence on social platforms — content strategy, posting, engagement, analytics.",
        "typical_skills": ["content strategy", "platform-specific best practices", "scheduling tools (Buffer, Hootsuite)", "analytics", "community engagement"],
        "typical_experience_years": 2,
        "typical_education": "BA in marketing, communications, or related",
        "typical_responsibilities": ["content calendars", "posting and engagement", "campaign reporting", "trend monitoring"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "sales_manager": {
        "esco_code": "S5.2.1", "preferred_label": "sales manager",
        "alt_labels": ["sales team lead", "regional sales manager"],
        "description": "Leads a team of sales representatives; owns quota, coaching, and pipeline.",
        "typical_skills": ["team leadership", "sales coaching", "pipeline management", "forecasting", "CRM tooling"],
        "typical_experience_years": 5,
        "typical_education": "BA in business or related; or strong sales track record",
        "typical_responsibilities": ["managing sales team", "quota planning", "deal review", "territory planning", "hiring sales reps"],
        "seniority_levels": ["mid", "senior", "director"],
    },
    "account_manager": {
        "esco_code": "S5.2.2", "preferred_label": "account manager",
        "alt_labels": ["client account manager", "key account manager"],
        "description": "Owns ongoing relationships with existing customers; drives renewals and expansion.",
        "typical_skills": ["relationship management", "consultative selling", "renewal management", "CRM", "stakeholder mapping"],
        "typical_experience_years": 3,
        "typical_education": "BA in business or related",
        "typical_responsibilities": ["customer health monitoring", "renewals", "expansion sales", "executive briefings"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "business_development_representative": {
        "esco_code": "S5.2.3", "preferred_label": "business development representative",
        "alt_labels": ["BDR", "sales development representative", "SDR"],
        "description": "Generates and qualifies sales leads; first stage of the outbound sales pipeline.",
        "typical_skills": ["outbound prospecting", "cold outreach", "CRM", "objection handling", "lead qualification (BANT/MEDDIC)"],
        "typical_experience_years": 1,
        "typical_education": "BA in business or related; or strong communication background",
        "typical_responsibilities": ["lead generation", "outbound calls/emails", "lead qualification", "meeting setting for AEs"],
        "seniority_levels": ["junior", "mid"],
    },

    # --- HR / Operations (4) ---
    "hr_specialist": {
        "esco_code": "S6.1.0", "preferred_label": "HR specialist",
        "alt_labels": ["human resources specialist", "people operations specialist"],
        "description": "Generalist HR role covering recruiting, onboarding, benefits, employee relations, and HR operations.",
        "typical_skills": ["HRIS systems", "labor law basics", "interview coordination", "onboarding processes", "employee relations"],
        "typical_experience_years": 2,
        "typical_education": "BA in HR, business, or psychology",
        "typical_responsibilities": ["recruiting support", "onboarding new hires", "benefits administration", "policy compliance", "employee inquiries"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "talent_acquisition_specialist": {
        "esco_code": "S6.1.1", "preferred_label": "talent acquisition specialist",
        "alt_labels": ["technical recruiter", "talent partner", "in-house recruiter"],
        "description": "Sources, screens, and hires candidates for open roles; partners with hiring managers.",
        "typical_skills": ["sourcing tools (LinkedIn Recruiter)", "ATS (Greenhouse, Lever)", "structured interviewing", "candidate assessment", "negotiation"],
        "typical_experience_years": 3,
        "typical_education": "BA in HR, business, or related",
        "typical_responsibilities": ["full-cycle recruiting", "candidate sourcing", "interview process design", "offer negotiation"],
        "seniority_levels": ["junior", "mid", "senior", "lead"],
    },
    "hr_business_partner": {
        "esco_code": "S6.1.2", "preferred_label": "HR business partner",
        "alt_labels": ["HRBP", "people business partner"],
        "description": "Strategic HR partner aligned to a business unit; advises leaders on people strategy.",
        "typical_skills": ["organizational design", "performance management", "employee relations", "change management", "data-driven HR"],
        "typical_experience_years": 6,
        "typical_education": "BA in HR or business; SHRM/CIPD certifications common",
        "typical_responsibilities": ["leadership coaching", "org design", "performance cycles", "succession planning", "employee relations escalations"],
        "seniority_levels": ["senior", "principal"],
    },
    "program_manager": {
        "esco_code": "S6.2.1", "preferred_label": "program manager",
        "alt_labels": ["technical program manager", "TPM (programs)", "senior project manager"],
        "description": "Coordinates large cross-team initiatives; tracks dependencies and drives delivery.",
        "typical_skills": ["program governance", "risk management", "cross-functional coordination", "executive reporting", "agile and waterfall fluency"],
        "typical_experience_years": 6,
        "typical_education": "BA/BSc plus PMP/PgMP common",
        "typical_responsibilities": ["program planning", "dependency tracking", "stakeholder communication", "risk mitigation", "release coordination"],
        "seniority_levels": ["senior", "principal"],
    },

    # --- Healthcare (3) ---
    "dentist": {
        "esco_code": "S7.1.1", "preferred_label": "dentist",
        "alt_labels": ["dental surgeon", "DDS"],
        "description": "Diagnoses and treats dental conditions; performs cleanings, fillings, extractions, and restorative procedures.",
        "typical_skills": ["clinical dentistry", "patient communication", "dental imaging interpretation", "infection control", "dental software"],
        "typical_experience_years": 4,
        "typical_education": "DDS or DMD; valid license",
        "typical_responsibilities": ["patient examinations", "treatment planning", "restorative work", "preventive care education"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "occupational_therapist": {
        "esco_code": "S7.1.2", "preferred_label": "occupational therapist",
        "alt_labels": ["OT", "rehabilitation therapist"],
        "description": "Helps patients recover or develop the skills needed for daily living and working after injury or illness.",
        "typical_skills": ["assessment of functional ability", "rehabilitation planning", "adaptive equipment knowledge", "patient education", "documentation"],
        "typical_experience_years": 2,
        "typical_education": "MSc/Doctorate in Occupational Therapy; license",
        "typical_responsibilities": ["patient evaluation", "treatment plans", "progress documentation", "home/work adaptation recommendations"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "dietitian": {
        "esco_code": "S7.1.3", "preferred_label": "dietitian",
        "alt_labels": ["registered dietitian", "nutritionist (clinical)"],
        "description": "Provides evidence-based nutritional counseling and medical nutrition therapy.",
        "typical_skills": ["clinical nutrition", "diet planning", "patient education", "EMR documentation", "diabetes/renal/cardiac diet specialization"],
        "typical_experience_years": 1,
        "typical_education": "BSc in Nutrition/Dietetics; RD credential",
        "typical_responsibilities": ["nutrition assessments", "individualized diet plans", "patient counseling", "interdisciplinary collaboration"],
        "seniority_levels": ["junior", "mid", "senior"],
    },

    # --- Creative (3) ---
    "illustrator": {
        "esco_code": "S8.1.1", "preferred_label": "illustrator",
        "alt_labels": ["digital illustrator", "editorial illustrator"],
        "description": "Creates original illustrations for editorial, advertising, publishing, or product use.",
        "typical_skills": ["digital illustration tools (Procreate, Illustrator)", "concept development", "color theory", "client briefs", "style adaptability"],
        "typical_experience_years": 2,
        "typical_education": "BFA in illustration or self-taught with strong portfolio",
        "typical_responsibilities": ["original artwork creation", "concept sketches", "revisions", "delivery in print/digital formats"],
        "seniority_levels": ["junior", "mid", "senior"],
    },
    "animator": {
        "esco_code": "S8.1.2", "preferred_label": "animator",
        "alt_labels": ["motion designer", "2D animator", "3D animator"],
        "description": "Brings characters, objects, and scenes to life through animation for film, games, or marketing.",
        "typical_skills": ["animation principles", "Maya/Blender/After Effects", "rigging or motion design", "storytelling through motion", "feedback iteration"],
        "typical_experience_years": 3,
        "typical_education": "BFA in animation or self-taught with portfolio",
        "typical_responsibilities": ["scene animation", "character animation", "motion design", "collaborating with directors"],
        "seniority_levels": ["junior", "mid", "senior", "lead"],
    },
    "art_director": {
        "esco_code": "S8.1.3", "preferred_label": "art director",
        "alt_labels": ["creative art director"],
        "description": "Leads visual direction for a project, brand, or campaign; oversees designers and illustrators.",
        "typical_skills": ["visual storytelling", "brand identity", "design leadership", "client presentation", "team management"],
        "typical_experience_years": 7,
        "typical_education": "BFA in design or related plus strong portfolio",
        "typical_responsibilities": ["creative direction", "design team management", "client/stakeholder communication", "quality assurance on visuals"],
        "seniority_levels": ["senior", "principal"],
    },
}


def main() -> None:
    raw = json.loads(ESCO_PATH.read_text(encoding="utf-8"))
    occupations = raw.get("occupations", {})
    before = len(occupations)

    skipped = []
    for key, entry in NEW_ROLES.items():
        if key in occupations:
            skipped.append(key)
            continue
        occupations[key] = entry

    after = len(occupations)

    if "_meta" in raw:
        raw["_meta"]["expanded"] = True
        raw["_meta"]["expansion_count"] = after - before

    ESCO_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"ESCO occupations: {before} -> {after} (+{after - before})")
    if skipped:
        print(f"Skipped (already present): {skipped}")
    print(f"Wrote: {ESCO_PATH}")


if __name__ == "__main__":
    main()
