# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export VERL_ZMQ_TMPDIR=/mnt/local2/wxy/tmp
mkdir -p /mnt/local2/wxy/tmp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CANDIDATE_NUM=${CANDIDATE_NUM:-}
# if [[ -n "${CANDIDATE_NUM}" ]]; then
#     DEFAULT_DATA_DIR=${PROJECT_ROOT}/data/tcm_herb_rerank_v3_shuffle_no_dup_c${CANDIDATE_NUM}
# else
#     DEFAULT_DATA_DIR=${PROJECT_ROOT}/data/tcm_herb_rerank_fullv2.1
# fi

# DEFAULT_DATA_DIR=${PROJECT_ROOT}/data/tcm_herb_rerank_c${CANDIDATE_NUM}_no_dup_filter_v10
DEFAULT_DATA_DIR=${PROJECT_ROOT}/data/tcm_herb_rerank_c${CANDIDATE_NUM}_v15
# DEFAULT_DATA_DIR=${PROJECT_ROOT}/data/tcm_herb_rerank_c${CANDIDATE_NUM}_v13

DATA_DIR=${DATA_DIR:-${DEFAULT_DATA_DIR}}
REWARD_PATH=${PROJECT_ROOT}/verl/rerank_reward.py
TCM_REWARD_STAGE=${TCM_REWARD_STAGE:-focus20}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_7b_tcm_rerank_${TCM_REWARD_STAGE}_fullv2_1}

# Resume template:
# - fresh run: RESUME_MODE=disable
# - auto resume same experiment: RESUME_MODE=auto
# - resume from explicit checkpoint:
#   RESUME_MODE=resume_path
#   RESUME_FROM_PATH=/abs/path/to/.../global_step_xxx
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/verl/checkpoints/verl_grpo_tcm_rerank/${EXPERIMENT_NAME}}

if [[ -n "${RESUME_FROM_PATH}" ]]; then
    RESUME_MODE=resume_path
fi

echo "Using reward stage: ${TCM_REWARD_STAGE}"
echo "Candidate num: ${CANDIDATE_NUM:-default}"
echo "Data dir: ${DATA_DIR}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Resume mode: ${RESUME_MODE}"
if [[ -n "${RESUME_FROM_PATH}" ]]; then
    echo "Resume from path: ${RESUME_FROM_PATH}"
fi
echo "Checkpoint dir: ${DEFAULT_LOCAL_DIR}"

CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_grpo_tcm_rerank' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@