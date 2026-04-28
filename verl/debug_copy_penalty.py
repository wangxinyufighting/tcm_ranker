import sys, json, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rerank_reward import compute_score, _get_reward_profile

with open("/mnt/local2/wxy/herb_rec+RLv2/outputs/test_prompt_response_qwen3_4b_sft4_v17_50.jsonl", "r") as f:
    lines = f.readlines()

total_copy_penalty = 0.0
for line in lines:
    data = json.loads(line)
    # reconstruct ground_truth 
    # wait, the jsonl output from test might not have the ground_truth dict directly, let's see its keys.
    pass

