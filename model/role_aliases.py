from typing import Dict

ROLE_ALIASES: Dict[str, str] = {
    # canonical snake_case -> canonical snake_case
    "conceptualization": "conceptualization",
    "methodology": "methodology",
    "software": "software",
    "validation": "validation",
    "formal_analysis": "formal_analysis",
    "investigation": "investigation",
    "resources": "resources",
    "data_curation": "data_curation",
    "writing_original_draft": "writing_original_draft",
    "writing_review_editing": "writing_review_editing",
    "visualization": "visualization",
    "supervision": "supervision",
    "project_administration": "project_administration",
    "funding_acquisition": "funding_acquisition",

    # Title Case / standard CRediT -> canonical snake_case
    "Conceptualization": "conceptualization",
    "Methodology": "methodology",
    "Software": "software",
    "Validation": "validation",
    "Formal Analysis": "formal_analysis",
    "Investigation": "investigation",
    "Resources": "resources",
    "Data Curation": "data_curation",
    "Writing - Original Draft": "writing_original_draft",
    "Writing – Original Draft": "writing_original_draft",
    "Writing - Review & Editing": "writing_review_editing",
    "Writing – Review & Editing": "writing_review_editing",
    "Visualization": "visualization",
    "Supervision": "supervision",
    "Project Administration": "project_administration",
    "Funding Acquisition": "funding_acquisition",
}