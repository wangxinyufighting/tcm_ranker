import sys, json, os, re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

with open("/mnt/local2/wxy/herb_rec+RLv2/outputs/test_prompt_response_qwen3_4b_sft4_v17_50.jsonl", "r") as f:
    lines = f.readlines()

total_sim_1 = 0.0
total_sim_2 = 0.0

for line in lines[:10]:
    data = json.loads(line)
    
    # parse candidates from prompt
    prompt = data["prompt"]
    match = re.search(r"候选中药（初排顺序）：(.*?)\\n", prompt)
    if match:
        cand_str = match.group(1).replace('、', '，')
        # wait, they use '、' or spaces in the real prompt
        cand_str = cand_str.replace(' ', '')
        candidates = [x.strip() for x in cand_str.split('、') if x.strip()]
    else:
        continue
        
    pred_raw = data.get("extracted_herbs", [])
    
    copy_num_min1 = len(pred_raw)
    base_top1 = candidates[:copy_num_min1]
    pred_top1 = pred_raw[:copy_num_min1]
    copy_count1 = sum(1 for p, b in zip(pred_top1, base_top1) if p == b)
    similarity1 = copy_count1 / copy_num_min1 if copy_num_min1 > 0 else 0.0
    
    print(f"Pred: {pred_top1[:3]}... Cand: {base_top1[:3]}... Copy count: {copy_count1}, Sim: {similarity1:.2f}")

