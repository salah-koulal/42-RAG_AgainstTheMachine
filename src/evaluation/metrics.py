"""Recall@k calculation used by the evaluate CLI command ."""

from typing import Any, Dict, List, Optional

from src.models import MinimalSource


class RecallFloat(float):
    """Float subclass containing hits and total question counts for evaluation."""

    hits: int
    total: int

    def __new__(cls, value: float, hits: int = 0, total: int = 0) -> "RecallFloat":
        obj = super().__new__(cls, value)
        obj.hits = hits
        obj.total = total
        return obj


def compute_iou(retrieved: MinimalSource, ground_truth: MinimalSource) -> float:
    """Compute 1D Intersection over Union (IoU) of character indices if files match."""
    if retrieved.file_path != ground_truth.file_path:
        return 0.0

    inter_start = max(retrieved.first_character_index, ground_truth.first_character_index)
    inter_end = min(retrieved.last_character_index, ground_truth.last_character_index)
    inter = max(0, inter_end - inter_start)
    if inter == 0:
        return 0.0

    union_start = min(retrieved.first_character_index, ground_truth.first_character_index)
    union_end = max(retrieved.last_character_index, ground_truth.last_character_index)
    union = union_end - union_start
    if union <= 0:
        return 0.0

    return inter / union


def sources_overlap(
    retrieved: MinimalSource,
    ground_truth: MinimalSource,
    min_iou: float = 0.05,
) -> bool:
    """Check if two sources refer to the same file and their character IoU >= min_iou."""
    return compute_iou(retrieved, ground_truth) >= min_iou


def question_hit(
    retrieved_sources: List[MinimalSource],
    ground_truth_sources: List[MinimalSource],
    min_iou: float = 0.05,
) -> bool:
    """Return True if at least one retrieved source overlaps with any ground truth source."""
    for gt in ground_truth_sources:
        for ret in retrieved_sources:
            if sources_overlap(ret, gt, min_iou=min_iou):
                return True
    return False


def compute_recall_at_k(
    student_results: List[Dict[str, Any]],
    ground_truth_questions: List[Dict[str, Any]],
    k: Optional[int] = None,
    min_iou: float = 0.05,
) -> RecallFloat:
    """Compute Recall@k across all ground-truth questions.

    Per the subject definition:
    - Overlap requires same file and IoU >= 0.05.
    - Question recall is the share of its correct sources retrieved in top-k.
    - Overall recall is the average over all ground-truth questions. Missing
      questions count as 0.0 recall.
    """
    total_questions = len(ground_truth_questions)
    if total_questions == 0:
        return RecallFloat(0.0, 0, 0)

    if k is not None and k <= 0:
        return RecallFloat(0.0, 0, total_questions)

    student_map: Dict[str, List[Dict[str, Any]]] = {}
    for result in student_results:
        qid = result.get("question_id")
        if qid:
            student_map[qid] = result.get("retrieved_sources", [])

    question_recalls: List[float] = []
    hit_questions = 0

    for q in ground_truth_questions:
        qid = q.get("question_id")
        gt_sources_raw = q.get("sources", [])
        if not gt_sources_raw:
            continue

        gt_sources = [MinimalSource(**s) for s in gt_sources_raw]
        raw_sources = student_map.get(qid, []) if qid else []
        if k is not None:
            raw_sources = raw_sources[:k]

        retrieved = [MinimalSource(**s) for s in raw_sources]

        # Calculate share of correct sources found
        found_sources = 0
        for gt in gt_sources:
            if any(sources_overlap(ret, gt, min_iou=min_iou) for ret in retrieved):
                found_sources += 1

        q_recall = found_sources / len(gt_sources)
        question_recalls.append(q_recall)
        if q_recall > 0:
            hit_questions += 1

    if not question_recalls:
        return RecallFloat(0.0, 0, total_questions)

    mean_recall = sum(question_recalls) / total_questions
    return RecallFloat(mean_recall, hit_questions, total_questions)
