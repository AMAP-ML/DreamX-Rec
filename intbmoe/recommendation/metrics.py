"""Pure-PyTorch ranking metrics for POI recommendation."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


RANK_CUTOFFS = (1, 2, 5)
CATEGORY_CUTOFFS = (1, 2, 5, 10)


def _selection_masks(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (all-except-newest, newest-only) masks for newest-first sessions."""

    newest = torch.zeros_like(valid)
    rows = valid.any(dim=1)
    if rows.any():
        row_indices = rows.nonzero(as_tuple=False).squeeze(-1)
        first_columns = valid[row_indices].to(torch.int64).argmax(dim=1)
        newest[row_indices, first_columns] = True
    return valid & ~newest, newest


def empty_metric_totals() -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for prefix in ("recommendation", "recommendation_newest"):
        totals[f"{prefix}_count"] = 0.0
        totals[f"{prefix}_category_count"] = 0.0
        for cutoff in RANK_CUTOFFS:
            totals[f"{prefix}_hr@{cutoff}_sum"] = 0.0
            totals[f"{prefix}_ndcg@{cutoff}_sum"] = 0.0
        for cutoff in CATEGORY_CUTOFFS:
            totals[f"{prefix}_category_inconsistency@{cutoff}_sum"] = 0.0
    return totals


@torch.no_grad()
def accumulate_recommendation_metrics(
    totals: Dict[str, float],
    logits: torch.Tensor,
    valid: torch.Tensor,
    positive_category: torch.Tensor,
    negative_category: torch.Tensor,
) -> None:
    """Accumulate HR/NDCG and category-inconsistency sufficient statistics."""

    history_mask, newest_mask = _selection_masks(valid.bool())
    candidate_category = torch.cat(
        [positive_category.unsqueeze(-1), negative_category], dim=-1
    )
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    sorted_categories = torch.gather(candidate_category, -1, sorted_indices)
    positive_rank = (sorted_indices == 0).to(torch.int64).argmax(dim=-1) + 1

    for prefix, selection in (
        ("recommendation", history_mask),
        ("recommendation_newest", newest_mask),
    ):
        count = int(selection.sum().item())
        totals[f"{prefix}_count"] += count
        if count == 0:
            continue
        ranks = positive_rank[selection]
        gains = 1.0 / torch.log2(ranks.to(torch.float64) + 1.0)
        for cutoff in RANK_CUTOFFS:
            hit = ranks <= min(cutoff, logits.size(-1))
            totals[f"{prefix}_hr@{cutoff}_sum"] += float(hit.sum().item())
            totals[f"{prefix}_ndcg@{cutoff}_sum"] += float((gains * hit).sum().item())

        selected_categories = sorted_categories[selection]
        selected_positive = positive_category[selection].unsqueeze(-1)
        category_valid = selected_positive.squeeze(-1) >= 0
        totals[f"{prefix}_category_count"] += int(category_valid.sum().item())
        for cutoff in CATEGORY_CUTOFFS:
            actual = min(cutoff, logits.size(-1))
            match = (selected_categories[:, :actual] == selected_positive).any(dim=-1)
            inconsistent = category_valid & ~match
            totals[f"{prefix}_category_inconsistency@{cutoff}_sum"] += float(
                inconsistent.sum().item()
            )


def finalize_recommendation_metrics(totals: Mapping[str, float]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for prefix in ("recommendation", "recommendation_newest"):
        count = totals[f"{prefix}_count"]
        category_count = totals[f"{prefix}_category_count"]
        metrics[f"{prefix}_count"] = count
        metrics[f"{prefix}_category_count"] = category_count
        denominator = max(1.0, count)
        category_denominator = max(1.0, category_count)
        for cutoff in RANK_CUTOFFS:
            metrics[f"{prefix}_hr@{cutoff}"] = totals[f"{prefix}_hr@{cutoff}_sum"] / denominator
            metrics[f"{prefix}_ndcg@{cutoff}"] = totals[f"{prefix}_ndcg@{cutoff}_sum"] / denominator
        for cutoff in CATEGORY_CUTOFFS:
            metrics[f"{prefix}_category_inconsistency@{cutoff}"] = (
                totals[f"{prefix}_category_inconsistency@{cutoff}_sum"] / category_denominator
            )
    return metrics
