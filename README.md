运行脚本：

MODEL_PATH=/mnt/local2/wxy/models/Qwen3-4B K=30 TCM_REWARD_STAGE=focus20 EXPERIMENT_NAME=qwen3_4b_C50_K30_rewardv40_no_sft  bash examples/grpo_trainer/run_qwen3-4b_tcm_herb_rerank_v2.sh

关注以下文件：


奖励函数： ./tcm_ranker/verl/rerank_reward.py

训练脚本：./tcm_ranker/verl/examples/grpo_trainer/run_qwen3-4b_tcm_herb_rerank_v2.sh

训练的模型存储在：./tcm_ranker/verl/checkpoints
