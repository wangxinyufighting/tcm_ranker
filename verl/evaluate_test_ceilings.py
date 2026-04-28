import sys
import os
import pandas as pd
import numpy as np

# Import functions from rerank_reward
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rerank_reward import _get_reward_profile, _safe_list, _collect_gt_variants, get_theoretical_optimal_score, _oracle_metrics_for_ranking, get_oracle_ranking

def main():
    df = pd.read_parquet('/mnt/local2/wxy/herb_rec+RLv2/data/tcm_herb_rerank_c30_no_dup_filter_v10/test.parquet')
    
    profile = _get_reward_profile()
    metric_weights = profile["metric_weights"]
    
    results = []
    
    for i, row in df.iterrows():
        ground_truth = row['reward_model'].get('ground_truth', {})
        if not isinstance(ground_truth, dict):
            continue
            
        candidate_herbs = _safe_list(ground_truth.get("candidate_herbs", []))
        candidate_set = set(candidate_herbs)
        gt_variants = _collect_gt_variants(ground_truth, candidate_set)
        
        if not candidate_herbs or not gt_variants:
            continue
            
        # 1. 最佳理论分数及匹配的 GT (best score phase)
        best_score, target_gt_variant = get_theoretical_optimal_score(gt_variants, candidate_herbs, metric_weights)
        eval_gt = [target_gt_variant]
        
        # 得到针对该 GT 的理想排序以获取 best_metrics 详情 (p10, r10, n10)
        best_ranking = get_oracle_ranking(eval_gt, candidate_herbs)
        best_metrics = _oracle_metrics_for_ranking(best_ranking, eval_gt)
        
        # 2. Base score 及 metrics (base score phase: 直接基于原有的 candidate_herbs 排序)
        base_metrics = _oracle_metrics_for_ranking(candidate_herbs, eval_gt)
        base_score = sum(w * base_metrics[m] for m, w in metric_weights.items())
        
        results.append({
            "idx": i,
            "best_score": best_score,
            "base_score": base_score,
            "best_p10": best_metrics.get("p10", 0.0),
            "best_r10": best_metrics.get("r10", 0.0),
            "best_n10": best_metrics.get("n10", 0.0),
            "base_p10": base_metrics.get("p10", 0.0),
            "base_r10": base_metrics.get("r10", 0.0),
            "base_n10": base_metrics.get("n10", 0.0)
        })
        
    res_df = pd.DataFrame(results)
    
    print("===== TEST SET CEILING & BASE ANALYSIS =====")
    print(f"Total valid samples: {len(res_df)}")
    if len(res_df) > 0:
        print(f"Average Best Score: {res_df['best_score'].mean():.4f}")
        print(f"Average Base Score: {res_df['base_score'].mean():.4f}")
        print(f"Best P@10/R@10/NDCG@10: {res_df['best_p10'].mean():.4f} / {res_df['best_r10'].mean():.4f} / {res_df['best_n10'].mean():.4f}")
        print(f"Base P@10/R@10/NDCG@10: {res_df['base_p10'].mean():.4f} / {res_df['base_r10'].mean():.4f} / {res_df['base_n10'].mean():.4f}")
    
    res_df.to_csv('/mnt/local2/wxy/herb_rec+RLv2/data/tcm_herb_rerank_c30_no_dup_filter_v10/test_scores_analysis.csv', index=False)
    print("\nDetailed results saved to: /mnt/local2/wxy/herb_rec+RLv2/data/tcm_herb_rerank_c30_no_dup_filter_v10/test_scores_analysis.csv")

if __name__ == "__main__":
    main()
