import os
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union
import numpy as np
from eval_herb_predictions import ndcg_at_k, dedup_preserve_order

# Reward target setup: align with eval_herb_predictions.py Oracle metrics.
# KS: Tuple[int, ...] = (5, 10, 20)
# KS: Tuple[int, ...] = (5, 10, 20, 30)
KS: Tuple[int, ...] = (5, 10, 20, 30)
KMAX: int = max(KS)

# Reward stage switch:
# - focus20: first phase, prioritize @20
# - full9: second phase, optimize all 9 metrics
DEFAULT_REWARD_STAGE = "focus20"
DEFAULT_REWARD_STAGE = "focus10"
DEFAULT_REWARD_STAGE = "focus5"
DEFAULT_REWARD_STAGE = "focus30"

REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "focus5": {
        "metric_weights": {
            "p5": 0.2,
            "r5": 0.5,
            "n5": 0.3,   
            "f1_5": 0.0,   
            
            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.0,
            "f1_10": 0.0, 
            
            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 0.0,
            
            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 10, 
    },
    
    "focus10": {
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.0,   
            "f1_5": 0.0,   
            
            "p10": 0.2,
            "r10": 0.5,
            "n10": 0.3,
            "f1_10": 0.0, 
            
            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 0.0,
            
            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,  # 现在我们只要求输出Top 10的核心药物即可
    },
    
    "focus20": {
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.0,   
            "f1_5": 0.0,   
            
            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.0,
            "f1_10": 0.0, 
            
            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 1.0,
            
            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 30,  # 现在我们只要求输出Top 10的核心药物即可
    }
}


def _get_reward_profile() -> Dict[str, Any]:
    stage = os.getenv("TCM_REWARD_STAGE", DEFAULT_REWARD_STAGE).strip().lower()
    if stage not in REWARD_PROFILES:
        stage = DEFAULT_REWARD_STAGE
    return REWARD_PROFILES[stage]


def extract_content(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_ranking_list(solution_str: str) -> List[str]:
    answer_content = extract_content(solution_str, "answer")
    if not answer_content:
        return []
    items = [part.strip() for part in answer_content.split(">")]
    return [x for x in items if x]


def check_response_format(solution_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.search(pattern, solution_str.strip(), re.DOTALL | re.IGNORECASE) else 0.0


def check_ranking_format(solution_str: str) -> float:
    answer_content = extract_content(solution_str, "answer")
    if not answer_content:
        return 0.0
    ranking_pattern = re.compile(r"^([^\s>]+(\s*>\s*[^\s>]+)*)$")
    return 1.0 if ranking_pattern.match(answer_content.strip()) else 0.0


def get_theoretical_optimal_metrics(gt_herbs_list: List[List[str]], candidate_herbs: List[str], metric_weights: Dict[str, float] = None) -> float:
    if metric_weights is None:
        metric_weights = _get_reward_profile()["metric_weights"]

    best_metrics = None
    best_score = 0.0
    
    for variant in gt_herbs_list:
        variant_set = []
        for i in variant:
            if i not in variant_set:
                variant_set.append(i)

        relevant_in_order = [h for h in variant_set if h in candidate_herbs]
        irrelevant_in_order = [h for h in candidate_herbs if h not in variant_set]
        ideal_ranking = relevant_in_order + irrelevant_in_order
        
        metrics = _oracle_metrics_for_ranking(ideal_ranking, [variant])
        score = sum(w * metrics.get(m, 0.0) for m, w in metric_weights.items())
        if score > best_score:
            best_metrics = metrics
            best_score = score

    return best_metrics


def get_theoretical_optimal_score(gt_herbs_list: List[List[str]], candidate_herbs: List[str], metric_weights: Dict[str, float] = None) -> float:
    if metric_weights is None:
        metric_weights = _get_reward_profile()["metric_weights"]

    best_score = 0.0
    
    for variant in gt_herbs_list:
        variant_set = []
        for i in variant:
            if i not in variant_set:
                variant_set.append(i)

        relevant_in_order = [h for h in variant_set if h in candidate_herbs]
        irrelevant_in_order = [h for h in candidate_herbs if h not in variant_set]
        ideal_ranking = relevant_in_order + irrelevant_in_order
        
        metrics = _oracle_metrics_for_ranking(ideal_ranking, [variant])
        score = sum(w * metrics.get(m, 0.0) for m, w in metric_weights.items())
        if score > best_score:
            best_score = score
            
    return best_score


def _oracle_metrics_for_ranking(ranking: Sequence[str], gt_herbs_list: Sequence[Sequence[str]]) -> Dict[str, float]:
    ranking = dedup_preserve_order(list(ranking))
    topk_max = ranking[:KMAX]

    metrics = {}
    for key in ["p", "r", "n", "f1_"]:
        for k in KS:
            metrics[f"{key}{k}"] = 0.0

    if not gt_herbs_list:
        return metrics

    for k in KS:
        topk = ranking[:k]
        best_p = 0.0
        best_r = 0.0
        best_n = 0.0

        for gt in gt_herbs_list:
            gt_set = set(gt)
            hits = sum(1 for h in topk if h in gt_set)
            p = hits / k
            r = hits / max(1, len(gt_set))
            if p > best_p:
                best_p = p
                best_r = r
                r_max = [1.0 if h in gt_set else 0.0 for h in topk_max]
                best_n = ndcg_at_k(r_max, k, method=1)

        metrics[f"p{k}"] = best_p
        metrics[f"r{k}"] = best_r
        metrics[f"n{k}"] = best_n
        
        divisor = (best_p + best_r)
        metrics[f"f1_{k}"] = (2 * best_p * best_r / divisor) if divisor > 0 else 0.0

    return metrics


def _get_k():
    stage = os.getenv("TCM_REWARD_STAGE", DEFAULT_REWARD_STAGE).strip().lower()
    if '20' in stage:
        return 20
    elif '10' in stage:
        return 10
    elif '5' in stage:
        return 5
    else:
        return 30
    

def compute_score(
    solution_str: str,
    ground_truth: Union[Dict[str, Any], List[Any]],
    data_source: str = "",
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    
    profile = _get_reward_profile()
    
    candidate_herbs = ground_truth["candidate_herbs"]
    gt_herbs_list = ground_truth["gt_herbs_list"]

    # 计算理论上限，并锁定“能够拿到最高上限得分的那一组GT”作为考核基准
    best_score = get_theoretical_optimal_score(gt_herbs_list, candidate_herbs, profile["metric_weights"])
    
    base_metrics = _oracle_metrics_for_ranking(candidate_herbs, gt_herbs_list)
    base_score = sum(w * base_metrics[m] for m, w in profile["metric_weights"].items())

    # ── 解析模型输出 ───────────────────────────────────────────
    pred_raw   = extract_ranking_list(solution_str)
    pred_dedup = dedup_preserve_order(pred_raw)
    
    # 提前计算 pred_metrics，因为后续监控指标一定需要它，防止 UnboundLocalError
    pred_metrics = _oracle_metrics_for_ranking(pred_dedup, gt_herbs_list)
    pred_score = sum(w * pred_metrics[m] for m, w in profile["metric_weights"].items())
    
    # ── 约束惩罚 ──────────────────────────────────────────────
    dup_penalty = 0
    invalid_penalty = 0
    shortfall_penalty = 0

    if not pred_raw:
        penalty = 1.0
    else:
        duplicate_count = max(0, len(pred_raw) - len(pred_dedup))
        invalid_count   = sum(1 for h in pred_dedup if h not in candidate_herbs)
        shortfall       = max(0, profile["min_output_items"] - len(pred_dedup))

        dup_penalty =       min(1, 0.03 * duplicate_count) if duplicate_count > 0 else 0.0
        invalid_penalty =   min(1, 0.03 * invalid_count)
        shortfall_penalty = min(1, 0.03 * shortfall)

        penalty = (
            dup_penalty
            + invalid_penalty
            + shortfall_penalty
        )

    # ── 格式奖励 ──────────────────────────────────────────────
    format_reward = (
        0.5 * check_response_format(solution_str)
        + 0.5 * check_ranking_format(solution_str)
    )

    # ── 质量奖励：相对提升 ────────────────────────────────────
    headroom = best_score - base_score
    
    if headroom < 0.05:
        quality_reward = 0.0  
    elif abs(pred_score - best_score) < 1e-6:
        quality_reward = 1.0
    else:
        quality_reward = (pred_score - base_score) / headroom
        quality_reward = max(-1.0, min(1.0, quality_reward))
        
        if abs(base_score - pred_score) < 0.01:
            quality_reward = -0.2

    # ── 汇总 ──────────────────────────────────────────────────
    
    penalty = min(penalty, 0.5)  # 总惩罚上限0.5分，防止过度惩罚导致训练不稳定
        
    if quality_reward < 0:
        final = quality_reward - penalty
    else:
        final = (
            quality_reward
            + profile["format_weight"] * format_reward
            - penalty
        )
        
    k_min = min(len(pred_dedup), len(candidate_herbs), KMAX)
    base_top_k = candidate_herbs[:k_min]
    pred_top_k = pred_dedup[:k_min]

    # 2. 位置原样抄袭惩罚 (Strict Position Copy)
    pos_match_count = sum(1 for p, b in zip(pred_top_k, base_top_k) if p == b)
    pos_similarity = pos_match_count / k_min if k_min > 0 else 0.0

    # ── 监控指标 ──────────────────────────────────────────────
    extra_info = {}
    for k in KS:
        for m in ["p", "r", "n"]:
            key = f"{m}{k}"
            extra_info[f"rerank_{key}"]  = pred_metrics[key]
            extra_info[f"prerank_{key}"] = base_metrics[key]
            extra_info[f"diff_{key}"]    = pred_metrics[key] - base_metrics[key]


    extra_info['format_reward'] = format_reward
    extra_info['quality_reward'] = quality_reward
    
    extra_info['invalid_penalty'] = invalid_penalty
    extra_info['shortfall_penalty'] = shortfall_penalty
    extra_info['penalty'] = penalty
    extra_info['pos_similarity'] = pos_similarity

    return {
        "score": final,
        **extra_info,
    }