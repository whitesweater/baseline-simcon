#!/bin/bash
# =============================================================================
# 三阶段训练脚本：Coconut + Decoder SIM-CoT
# 
# 实验设置：
#   c_thought: 6
#   epochs_per_stage: 1  
#   max_latent_stage: 1
#
# 训练流程：
#   Stage 1: LLM-only Training - 隐式 token 参与训练，逐渐移除显式推理步骤
#            期望性能 ~39
#   Stage 2: Decoder-only Training - 冻结 LLM，只训练 decoder
#            性能保持不变
#   Stage 3: Joint Training - LLM 和 decoder 联合训练
#            期望性能 ~42
# =============================================================================

set -e  # 遇到错误立即退出

# 配置 GPU 数量
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_stage() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

echo_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# =============================================================================
# Stage 1: LLM-only Training
# =============================================================================
run_stage1() {
    echo_stage "Stage 1: LLM-only Training"
    echo "在此阶段，所有定义的隐式 token 从一开始就参与训练"
    echo "显式推理步骤逐渐被移除"
    echo "期望性能: ~39"
    echo ""
    
    torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        run.py args/gsm_3stage_stage1_llm_only.yaml
    
    echo_stage "Stage 1 完成！"
    echo "Checkpoint 保存在: ./ckpts/gsm_3stage_stage1_llm_only/"
}

# =============================================================================
# Stage 2: Decoder-only Training
# =============================================================================
run_stage2() {
    echo_stage "Stage 2: Decoder-only Training"
    echo "在此阶段，只训练 decoder"
    echo "LLM 被冻结，模型性能保持不变"
    echo ""
    
    # 检查 Stage 1 checkpoint 是否存在
    if [ ! -f "./ckpts/gsm_3stage_stage1_llm_only/checkpoint_3" ]; then
        echo_error "Stage 1 checkpoint 不存在！请先运行 Stage 1"
        echo "期望路径: ./ckpts/gsm_3stage_stage1_llm_only/checkpoint_3"
        exit 1
    fi
    
    torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        run.py args/gsm_3stage_stage2_decoder_only.yaml
    
    echo_stage "Stage 2 完成！"
    echo "Checkpoint 保存在: ./ckpts/gsm_3stage_stage2_decoder_only/"
}

# =============================================================================
# Stage 3: Joint Training
# =============================================================================
run_stage3() {
    echo_stage "Stage 3: Joint Training of Coconut and Decoder"
    echo "在此阶段，LLM 和 decoder 联合训练"
    echo "期望性能: ~42"
    echo ""
    
    # 检查 Stage 2 checkpoint 是否存在
    if [ ! -f "./ckpts/gsm_3stage_stage2_decoder_only/checkpoint_3" ]; then
        echo_error "Stage 2 checkpoint 不存在！请先运行 Stage 2"
        echo "期望路径: ./ckpts/gsm_3stage_stage2_decoder_only/checkpoint_3"
        exit 1
    fi
    
    torchrun --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        run.py args/gsm_3stage_stage3_joint.yaml
    
    echo_stage "Stage 3 完成！"
    echo "最终 Checkpoint 保存在: ./ckpts/gsm_3stage_stage3_joint/"
}

# =============================================================================
# 完整三阶段训练
# =============================================================================
run_all() {
    echo_stage "开始完整三阶段训练"
    
    run_stage1
    echo ""
    
    run_stage2
    echo ""
    
    run_stage3
    
    echo_stage "三阶段训练全部完成！"
    echo "Stage 1 (LLM-only):     ./ckpts/gsm_3stage_stage1_llm_only/"
    echo "Stage 2 (Decoder-only): ./ckpts/gsm_3stage_stage2_decoder_only/"
    echo "Stage 3 (Joint):        ./ckpts/gsm_3stage_stage3_joint/"
}

# =============================================================================
# 主入口
# =============================================================================
usage() {
    echo "Usage: $0 [stage1|stage2|stage3|all]"
    echo ""
    echo "Options:"
    echo "  stage1  - 运行 Stage 1: LLM-only Training"
    echo "  stage2  - 运行 Stage 2: Decoder-only Training"
    echo "  stage3  - 运行 Stage 3: Joint Training"
    echo "  all     - 依次运行所有三个阶段"
    echo ""
    echo "Environment variables:"
    echo "  NPROC_PER_NODE - GPU 数量 (默认: 8)"
    echo ""
    echo "Example:"
    echo "  NPROC_PER_NODE=4 $0 all"
}

case "${1:-all}" in
    stage1)
        run_stage1
        ;;
    stage2)
        run_stage2
        ;;
    stage3)
        run_stage3
        ;;
    all)
        run_all
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo_error "未知参数: $1"
        usage
        exit 1
        ;;
esac
