"""Recall@k calculation used by the evaluate CLI command ."""

from typing import Any, Dict, List

from src.models import MinimalSource


def sources_overlap(retrieved: MinimalSource, ground_truth: MinimalSource) -> bool:
    """Check if two sources refer to the same file and their character ranges overlap."""
    if retrieved.file_path != ground_truth.file_path:
        return False
    return (
        retrieved.first_character_index < ground_truth.last_character_index
        and ground_truth.first_character_index < retrieved.last_character_index
    )


def question_hit(
    retrieved_sources: List[MinimalSource],
    ground_truth_sources: List[MinimalSource],
) -> bool:
    """Return True if at least one retrieved source overlaps with any ground truth source."""
    for ret in retrieved_sources:
        for gt in ground_truth_sources:
            if sources_overlap(ret, gt):
                return True
    return False


def compute_recall_at_k(
    student_results: List[Dict[str, Any]],
    ground_truth_questions: List[Dict[str, Any]],
) -> float:
    """Compute Recall@k across all questions, returning a float between 0.0 and 1.0."""
    gt_map: Dict[str, List[MinimalSource]] = {}
    for q in ground_truth_questions:
        gt_map[q["question_id"]] = [MinimalSource(**s) for s in q["sources"]]

    hits = 0
    total = 0

    for result in student_results:
        qid = result["question_id"]
        if qid not in gt_map:
            continue

        retrieved = [MinimalSource(**s) for s in result["retrieved_sources"]]
        gt_sources = gt_map[qid]
        total += 1

        if question_hit(retrieved, gt_sources):
            hits += 1

    if total == 0:
        return 0.0
    return hits / total
