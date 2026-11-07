import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

DOCS_DIR = r"D:\ПРОФ\ВУЗ\ВКР\docs"

# Define destination subdirectories
dirs_to_create = [
    os.path.join(DOCS_DIR, "guidelines"),
    os.path.join(DOCS_DIR, "specifications"),
    os.path.join(DOCS_DIR, "literature"),
    os.path.join(DOCS_DIR, "datasets"),
    os.path.join(DOCS_DIR, "models"),
    os.path.join(DOCS_DIR, "benchmarks"),
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"Ensured directory: {d}")

# Mapping of old root files to new locations
file_moves = [
    # Guidelines & Protocols
    ("00_system_and_agent_guidelines.md", os.path.join("guidelines", "system_and_agent_guidelines.md")),
    ("incident_synthetic_metrics_prohibition.md", os.path.join("guidelines", "prohibition_synthetic_metrics.md")),
    ("meta_loop_verification.md", os.path.join("guidelines", "meta_loop_verification.md")),

    # Specifications
    ("01_research_topic_and_aspects.md", os.path.join("specifications", "research_topic_and_scope.md")),
    ("02_metrics_and_losses.md", os.path.join("specifications", "metrics_and_losses.md")),
    ("03_hypotheses.md", os.path.join("specifications", "hypotheses_registry.md")),
    ("08_task_specification_mvp_instruction.md", os.path.join("specifications", "mvp_task_specification.md")),

    # Literature & SOTA
    ("04_sota_review_2026.md", os.path.join("literature", "sota_review.md")),
    ("06_papers_40plus.md", os.path.join("literature", "bibliography_papers.md")),
    ("07_github_projects.md", os.path.join("literature", "github_repositories.md")),

    # Datasets
    ("05_datasets.md", os.path.join("datasets", "catalog.md")),

    # Models & Architectures
    ("09_ready_models_and_aggregators.md", os.path.join("models", "ready_models_and_aggregators.md")),
    ("10_architectural_approaches_gaze_and_facial_gestures.md", os.path.join("models", "architectural_approaches_gaze_gestures.md")),
    ("11_detailed_model_architectures.md", os.path.join("models", "detailed_model_architectures.md")),

    # Benchmarks & Leaderboards
    ("leaderboard.csv", os.path.join("benchmarks", "leaderboard.csv")),
    ("leaderboard.tsv", os.path.join("benchmarks", "leaderboard.tsv")),
]

# Move existing files if present
for src_name, dst_rel in file_moves:
    src_path = os.path.join(DOCS_DIR, src_name)
    dst_path = os.path.join(DOCS_DIR, dst_rel)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_name} -> {dst_rel}")
    else:
        print(f"Notice: {src_name} already moved or not present.")

print("Reorganization of existing files complete.")
