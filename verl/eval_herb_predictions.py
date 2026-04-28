from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def norm_sym(sym_str: str) -> str:
    return " ".join(sym_str.strip().split())


def parse_sym_herbs_line(line: str, allow_empty_herbs: bool = False) -> Tuple[str, List[int]] | None:
    """
    解析症状-中药数据行
    输入格式: "症状1 症状2\t herb_id1 herb_id2 herb_id3"
    
    Args:
        line: 文件中的一行数据
    
    Returns:
        (症状字符串, [中药ID列表]) 或 None（空行或格式错误）
        
    Examples:
        "发热 咳嗽\t1 2 3 4 5" -> ("发热 咳嗽", [1, 2, 3, 4, 5])
    """
    line = line.strip("\n")
    if not line.strip() or "\t" not in line:
        return None
    left, right = line.split("\t", 1)
    left = norm_sym(left)
    herbs = [int(x) for x in right.strip().split() if x]
    if not left:
        return None
    if not herbs and not allow_empty_herbs:
        return None
    return left, herbs


def dedup_preserve_order(xs: Sequence[int]) -> List[int]:
    """
    去重并保持原始顺序
    用于去除预测列表中的重复ID，同时保持排名顺序
    
    Args:
        xs: 可能包含重复的ID序列
    
    Returns:
        去重后的ID列表（保持首次出现顺序）
        
    Examples:
        [1, 2, 1, 3, 2] -> [1, 2, 3]
    """
    seen = set()
    out: List[int] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@dataclass
class Metrics:
    precision: Dict[int, float]
    recall: Dict[int, float]
    ndcg: Dict[int, float]
    rmrr: Dict[int, float]


def dcg_at_k(r: Sequence[float], k: int, method: int = 1) -> float:
    """
    Match utils/batch_test.py:dcg_at_k behavior (method=1 default).
    计算DCG@k (Discounted Cumulative Gain at k)
    衡量排序质量的指标，考虑位置折扣
    
    DCG公式:
    - method=0: DCG = r[0] + Σ(i=1..k) r[i] / log2(i+1)
    - method=1: DCG = Σ(i=0..k-1) r[i] / log2(i+2)  [默认，更常用]
    
    Args:
        r: 相关性分数列表 (1.0表示相关, 0.0表示不相关)
        k: 考虑前k个结果
        method: 计算方法 (0或1)
    
    Returns:
        DCG@k值
    
    Examples:
        dcg_at_k([1, 1, 0, 1], 3, method=1)
        = 1/log2(2) + 1/log2(3) + 0/log2(4)
        = 1 + 0.63 + 0 = 1.63
    """
    rr = list(r)[:k]
    if not rr:
        return 0.0
    if method == 0:
        # r[0] + sum_{i=1..} r[i] / log2(i+1)
        out = float(rr[0])
        for i in range(1, len(rr)):
            out += float(rr[i]) / math.log2(i + 1)
        return out
    if method == 1:
        # sum_{i=0..} r[i] / log2(i+2)
        out = 0.0
        for i, rel in enumerate(rr):
            out += float(rel) / math.log2(i + 2)
        return out
    raise ValueError("method must be 0 or 1")


def ndcg_at_k(r: Sequence[float], k: int, method: int = 1) -> float:
    """Match utils/batch_test.py:ndcg_at_k behavior."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def ndcg_standard_at_k(r: Sequence[float], k: int, n_relevant_total: int, method: int = 1) -> float:
    """Standard NDCG@k using an ideal ranking with min(k, |GT|) relevant items.

    Unlike utils/batch_test.py behavior, the IDCG denominator here does NOT depend
    on how many relevant items happen to appear within a longer K_max list.
    """
    if k <= 0 or n_relevant_total <= 0:
        return 0.0
    dcg = dcg_at_k(r, k, method)
    ideal_ones = min(k, n_relevant_total)
    idcg = dcg_at_k([1.0] * ideal_ones, k, method)
    if not idcg:
        return 0.0
    return dcg / idcg


def rank_mrr_at_k(gt_herbs_ordered: Sequence[int], pred_rank: Sequence[int], k: int) -> float:
    """Match utils/batch_test.py repeat==1 Rank-MRR (rmrr) definition.

    rmrr@k = (1/|GT|) * sum_{a_rank in [0..|GT|-1], gt in topk}
                1 / (|a_refer - a_rank| + 1)
    where a_rank is the GT list index and a_refer is the index in the top-k list.
    """
    if not gt_herbs_ordered:
        return 0.0
    topk = list(pred_rank)[:k]
    score = 0.0
    for a_rank, herb in enumerate(gt_herbs_ordered):
        if herb in topk:
            a_refer = topk.index(herb)
            score += 1.0 / (abs(a_refer - a_rank) + 1)
    return score / len(gt_herbs_ordered)


def eval_strict_per_line(
    gt_lines: List[Tuple[str, List[int]]],
    pred_lines: List[Tuple[str, List[int]]],
    ks: Iterable[int],
    *,
    ndcg_kmax: int | None = None,
    ndcg_mode: str = "batch_test",
) -> Tuple[Metrics, int]:
    ks = list(ks)
    k_needed = max(ks) if ks else 0
    k_max = ndcg_kmax if ndcg_kmax is not None else k_needed
    if k_max < k_needed:
        raise ValueError(f"ndcg_kmax must be >= max(ks)={k_needed}, got {k_max}")
    p_sum = {k: 0.0 for k in ks}
    r_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    rmrr_sum = {k: 0.0 for k in ks}

    mismatched = 0
    n = min(len(gt_lines), len(pred_lines))
    for i in range(n):
        gt_sym, gt_herbs = gt_lines[i]
        pr_sym, pr_rank = pred_lines[i]
        if gt_sym != pr_sym:
            mismatched += 1
        gt_set = set(gt_herbs)
        pr_rank = dedup_preserve_order(pr_rank)

        # NDCG modes:
        # - batch_test: build r at top-K_max (max(ks) by default), then ndcg_at_k(r, k)
        #              (this can make NDCG@k change when you change K_max)
        # - hits:      build r at top-k for each k, then ndcg_at_k(r_k, k)
        # - standard:  build r at top-k and normalize by ideal min(k, |GT|)
        if ndcg_mode == "batch_test":
            topk_max = pr_rank[:k_max]
            r_max = [1.0 if h in gt_set else 0.0 for h in topk_max]
        elif ndcg_mode in {"hits", "standard"}:
            r_max = []
        else:
            raise ValueError(f"Unknown ndcg_mode={ndcg_mode!r}. Use one of: batch_test, hits, standard")

        for k in ks:
            topk = pr_rank[:k]
            hits = sum(1 for h in topk if h in gt_set)
            p_sum[k] += hits / k
            r_sum[k] += hits / max(1, len(gt_set))
            if ndcg_mode == "batch_test":
                ndcg_sum[k] += ndcg_at_k(r_max, k, method=1)
            elif ndcg_mode == "hits":
                r_k = [1.0 if h in gt_set else 0.0 for h in topk]
                ndcg_sum[k] += ndcg_at_k(r_k, k, method=1)
            else:  # standard
                r_k = [1.0 if h in gt_set else 0.0 for h in topk]
                ndcg_sum[k] += ndcg_standard_at_k(r_k, k, n_relevant_total=len(gt_set), method=1)
            rmrr_sum[k] += rank_mrr_at_k(gt_herbs, pr_rank, k)

    denom = float(n) if n else 1.0
    return Metrics(
        precision={k: p_sum[k] / denom for k in ks},
        recall={k: r_sum[k] / denom for k in ks},
        ndcg={k: ndcg_sum[k] / denom for k in ks},
        rmrr={k: rmrr_sum[k] / denom for k in ks},
    ), mismatched


def eval_oracle_multi_gt(
    gt_lines: List[Tuple[str, List[int]]],
    pred_lines: List[Tuple[str, List[int]]],
    ks: Iterable[int],
    *,
    ndcg_kmax: int | None = None,
    ndcg_mode: str = "batch_test",
) -> Tuple[Metrics, int]:
    """
    Oracle多GT评估模式
    - 同一症状可能有多个有效的真实处方
    - 对于每个预测，选择能够最大化P@k的真实处方进行评估
    - 这是"最佳情况"评估，给出模型性能的上界
    
    工作原理:
    1. 收集所有相同症状的GT处方
    2. 对于每个k值，遍历所有可能的GT处方
    3. 选择使Precision@k最大的那个处方
    4. 使用该处方的Recall@k、NDCG@k等指标
    
    Args:
        gt_lines: 真实标签列表
        pred_lines: 预测结果列表
        ks: 评估的k值列表
        ndcg_kmax: NDCG计算时使用的最大k值
        ndcg_mode: NDCG计算模式
        
    Returns:
        (Metrics对象, 不匹配的行数)
        
    Example:
        症状"发热"可能有多个有效处方:
        - [1, 2, 3]
        - [1, 4, 5]
        如果预测是[1, 4, 6, 7], 对k=2:
        - 与[1,2,3]比较: hits=1, P@2=0.5
        - 与[1,4,5]比较: hits=2, P@2=1.0 ✓ (选择这个)
    """
    ks = list(ks)
    k_needed = max(ks) if ks else 0
    k_max = ndcg_kmax if ndcg_kmax is not None else k_needed
    if k_max < k_needed:
        raise ValueError(f"ndcg_kmax must be >= max(ks)={k_needed}, got {k_max}")
    sym_to_gt_sets: Dict[str, List[List[int]]] = defaultdict(list)
    for sym, herbs in gt_lines:
        sym_to_gt_sets[sym].append(herbs)

    p_sum = {k: 0.0 for k in ks}
    r_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    rmrr_sum = {k: 0.0 for k in ks}

    mismatched = 0
    n = min(len(gt_lines), len(pred_lines))
    for i in range(n):
        gt_sym, _ = gt_lines[i]
        pr_sym, pr_rank = pred_lines[i]
        if gt_sym != pr_sym:
            mismatched += 1
        v_list = sym_to_gt_sets[gt_sym]
        pr_rank = dedup_preserve_order(pr_rank)

        for k in ks:
            topk = pr_rank[:k]
            best_p = 0.0
            best_r = 0.0
            best_ndcg = 0.0
            best_rmrr = 0.0
            for v in v_list:
                v_set = set(v)
                hits = sum(1 for h in topk if h in v_set)
                p = hits / k
                r = hits / max(1, len(v_set))
                if p > best_p:
                    best_p = p
                    best_r = r
                    if ndcg_mode == "batch_test":
                        topk_max = pr_rank[:k_max]
                        r_max = [1.0 if h in v_set else 0.0 for h in topk_max]
                        best_ndcg = ndcg_at_k(r_max, k, method=1)
                    elif ndcg_mode == "hits":
                        r_k = [1.0 if h in v_set else 0.0 for h in topk]
                        best_ndcg = ndcg_at_k(r_k, k, method=1)
                    elif ndcg_mode == "standard":
                        r_k = [1.0 if h in v_set else 0.0 for h in topk]
                        best_ndcg = ndcg_standard_at_k(r_k, k, n_relevant_total=len(v_set), method=1)
                    else:
                        raise ValueError(
                            f"Unknown ndcg_mode={ndcg_mode!r}. Use one of: batch_test, hits, standard"
                        )
                    best_rmrr = rank_mrr_at_k(v, pr_rank, k)
            p_sum[k] += best_p
            r_sum[k] += best_r
            ndcg_sum[k] += best_ndcg
            rmrr_sum[k] += best_rmrr

    denom = float(n) if n else 1.0
    return Metrics(
        precision={k: p_sum[k] / denom for k in ks},
        recall={k: r_sum[k] / denom for k in ks},
        ndcg={k: ndcg_sum[k] / denom for k in ks},
        rmrr={k: rmrr_sum[k] / denom for k in ks},
    ), mismatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, default="data/test_gt_id.txt")
    ap.add_argument("--pred", type=str, default="test_0224.txt")
    ap.add_argument("--ks", type=str, default="5,10,15,20")
    ap.add_argument(
        "--ndcg_mode",
        type=str,
        default="batch_test",
        choices=["batch_test", "hits", "standard"],
        help=(
            "How to compute NDCG. "
            "batch_test: replicate utils/batch_test.py (IDCG depends on K_max=max(ks) unless --ndcg_kmax is set); "
            "hits: compute IDCG from hits within top-k only; "
            "standard: compute IDCG assuming min(k, |GT|) ideal relevant items." 
        ),
    )
    ap.add_argument(
        "--ndcg_kmax",
        type=int,
        default=None,
        help="Compute NDCG using relevance vector built at top-K_max. "
        "Default: max(ks) (matches utils/batch_test.py).",
    )
    ap.add_argument(
        "--allow_empty_pred_herbs",
        action="store_true",
        help="Keep prediction lines with empty herb list to avoid line-index shift during evaluation.",
    )
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    gt_lines: List[Tuple[str, List[int]]] = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_sym_herbs_line(line, allow_empty_herbs=False)
            if parsed:
                gt_lines.append(parsed)

    pred_lines: List[Tuple[str, List[int]]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_sym_herbs_line(line, allow_empty_herbs=args.allow_empty_pred_herbs)
            if parsed:
                pred_lines.append(parsed)

    n = min(len(gt_lines), len(pred_lines))
    print(f"GT lines={len(gt_lines)}  Pred lines={len(pred_lines)}  Using n={n}")

    if args.ndcg_mode == "batch_test" and args.ndcg_kmax is None and len(ks) > 1:
        print(
            "NOTE: ndcg_mode=batch_test uses K_max=max(ks). "
            "If you change ks across runs, NDCG@small-k may change too. "
            "To compare across runs, keep ks fixed, set --ndcg_kmax to a fixed value, "
            "or use --ndcg_mode standard."
        )

    strict, mismatch_strict = eval_strict_per_line(
        gt_lines,
        pred_lines,
        ks,
        ndcg_kmax=args.ndcg_kmax,
        ndcg_mode=args.ndcg_mode,
    )
    oracle, mismatch_oracle = eval_oracle_multi_gt(
        gt_lines,
        pred_lines,
        ks,
        ndcg_kmax=args.ndcg_kmax,
        ndcg_mode=args.ndcg_mode,
    )

    print("\n[Strict per-line GT]")
    print(f"symptom-key mismatches (by line index): {mismatch_strict}")
    for k in ks:
        print(
            f"P@{k}={strict.precision[k]:.4f}  R@{k}={strict.recall[k]:.4f}  "
            f"NDCG@{k}={strict.ndcg[k]:.4f}  rank_mrr@{k}={strict.rmrr[k]:.4f}"
        )

    print("\n[Oracle multi-GT per symptom-set (matches batch_test.test selection rule)]")
    print(f"symptom-key mismatches (by line index): {mismatch_oracle}")
    for k in ks:
        print(
            f"P@{k}={oracle.precision[k]:.4f}  R@{k}={oracle.recall[k]:.4f}  "
            f"NDCG@{k}={oracle.ndcg[k]:.4f}  rank_mrr@{k}={oracle.rmrr[k]:.4f}"
        )


if __name__ == "__main__":
    main()
