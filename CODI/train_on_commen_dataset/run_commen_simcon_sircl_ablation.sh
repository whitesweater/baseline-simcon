#!/bin/bash
# =============================================================================
# commonsense 数据集 simcon_sircl 模型 - 灵活训练/测试脚本
# =============================================================================
# 支持任意 trajectory_loss_factor 值的训练和测试
#
# 用法:
#   ./run_commen_simcon_sircl_ablation.sh train -f 0.25     # 训练 factor=0.25
#   ./run_commen_simcon_sircl_ablation.sh test -f 0.1       # 测试 factor=0.1
#   ./run_commen_simcon_sircl_ablation.sh all -f 0.25       # 训练+测试
#   ./run_commen_simcon_sircl_ablation.sh list              # 列出所有已有模型
#   ./run_commen_simcon_sircl_ablation.sh test -f baseline  # 测试 baseline
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# 加载环境配置
source "${PROJECT_DIR}/config.env" || { echo "Error: config.env not found"; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# 基础配置
RESULT_DIR="${CODI_RESULT_DIR}/commen_ablation"
TRAINED_DIR="${CODI_SAVE_DIR}"
BASELINE_NAME="commen_simcon_base"
BASELINE_PATH="${TRAINED_DIR}/commen_simcon_base"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 根据 factor 获取模型名和路径
# =============================================================================
get_model_info() {
    local factor="$1"
    
    case "$factor" in
        "baseline")
            MODEL_NAME="commen_simcon_base"
            MODEL_PATH="${BASELINE_PATH}"
            ;;
        "0.1")
            # 特殊处理：已有的 0.1 模型
            MODEL_NAME="commen_simcon_sircl"
            MODEL_PATH="${TRAINED_DIR}/commen_simcon_sircl"
            ;;
        *)
            # 通用命名：将小数点替换为空，如 0.05 -> factor005, 0.25 -> factor025
            local factor_suffix=$(echo "$factor" | sed 's/0\.//' | sed 's/\.//')
            MODEL_NAME="commen_simcon_sircl_factor${factor_suffix}"
            MODEL_PATH="${TRAINED_DIR}/${MODEL_NAME}"
            ;;
    esac
}

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  train      训练模型"
    echo "  test       测试模型"
    echo "  all        训练 + 测试"
    echo "  list       列出所有已有模型"
    echo ""
    echo "选项:"
    echo "  -f, --factor F   指定 trajectory_loss_factor 值 (必须)"
    echo "                   可用: 任意浮点数 或 'baseline'"
    echo "  --dry-run        预览命令，不实际执行"
    echo "  -h, --help       显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 train -f 0.25          # 训练 factor=0.25"
    echo "  $0 test -f 0.1            # 测试 factor=0.1"
    echo "  $0 all -f 0.15            # 训练+测试 factor=0.15"
    echo "  $0 test -f baseline       # 测试 baseline"
    echo "  $0 list                   # 列出所有模型"
    echo ""
    echo "已有模型:"
    list_models
    exit 0
}

# =============================================================================
# 列出已有模型
# =============================================================================
list_models() {
    echo ""
    echo "  Baseline: ${BASELINE_PATH}"
    if [[ -d "$BASELINE_PATH" ]]; then
        echo "    状态: ✓"
    else
        echo "    状态: ✗ 不存在"
    fi
    echo ""
    echo "  消融模型 (outputs 目录):"
    for dir in "${TRAINED_DIR}"/commen_simcon_sircl*; do
        if [[ -d "$dir" ]]; then
            local name=$(basename "$dir")
            echo "    ✓ $name"
        fi
    done 2>/dev/null || echo "    (无)"
}

# =============================================================================
# 动态生成训练命令
# =============================================================================
run_training() {
    local factor="$1"
    
    if [[ "$factor" == "baseline" ]]; then
        log_info "Baseline 使用单独的训练脚本，跳过"
        return 0
    fi
    
    get_model_info "$factor"
    
    log_info "训练模型: ${MODEL_NAME}"
    log_info "Factor: ${factor}"
    log_info "输出目录: ${MODEL_PATH}"
    
    if [[ -d "$MODEL_PATH" ]]; then
        log_warn "模型已存在: $MODEL_PATH"
        read -p "是否覆盖? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "跳过训练"
            return 0
        fi
    fi
    
    # 动态选择端口避免冲突
    local port=$((22620 + RANDOM % 100))
    
    train_cmd="torchrun --nnodes 1 --master_port ${port} --nproc_per_node 2 ${PROJECT_DIR}/train.py \
        --output_dir \"${TRAINED_DIR}\" \
        --expt_name ${MODEL_NAME} \
        --logging_dir \"${TRAINED_DIR}/logs/${MODEL_NAME}-logs\" \
        --logging_steps 10 \
        --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
        --data_name commonsense \
        --seed 11 \
        --model_max_length 512 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --bf16 \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory True \
        --dataloader_persistent_workers True \
        --dataloader_prefetch_factor 2 \
        --num_train_epochs 30 \
        --learning_rate 8e-4 \
        --max_grad_norm 2.0 \
        --use_lora True \
        --lora_r 128 \
        --lora_alpha 32 \
        --lora_init \
        --save_strategy epoch \
        --save_total_limit 200 \
        --save_safetensors False \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --do_train \
        --report_to tensorboard \
        --num_latent 6 \
        --logging_strategy steps \
        --use_prj True \
        --prj_dim 2048 \
        --prj_dropout 0.0 \
        --distill_loss_div_std True \
        --exp_mode False \
        --exp_data_num 200 \
        --remove_eos True \
        --distill_loss_factor 20 \
        --print_ref_model_stats False \
        --max_token_num 200 \
        --use_decoder True \
        --use_trajectory_consistency True \
        --trajectory_space_type euclidean \
        --trajectory_radius_threshold 2 \
        --trajectory_loss_factor ${factor}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $train_cmd"
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        eval "$train_cmd"
        
        if [[ $? -eq 0 ]]; then
            log_success "训练完成: ${MODEL_NAME}"
        else
            log_error "训练失败: ${MODEL_NAME}"
            return 1
        fi
    fi
}

# =============================================================================
# 测试函数
# =============================================================================
run_testing() {
    local factor="$1"
    
    get_model_info "$factor"
    
    log_info "测试模型: ${MODEL_NAME}"
    log_info "Checkpoint: ${MODEL_PATH}"
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "模型不存在: ${MODEL_PATH}"
        log_info "请先运行: $0 train -f ${factor}"
        return 1
    fi
    
    mkdir -p "${RESULT_DIR}"
    
    test_cmd="python ${PROJECT_DIR}/test_multi_dataset.py \
        --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
        --ckpt_dir \"${MODEL_PATH}\" \
        --datasets \"commonsense\" \
        --num_runs 1 \
        --result_dir \"${RESULT_DIR}\" \
        --seed 11 \
        --model_max_length 512 \
        --bf16 \
        --lora_r 128 --lora_alpha 32 --lora_init \
        --batch_size 32 \
        --greedy True \
        --num_latent 6 \
        --use_prj True \
        --prj_dim 2048 \
        --prj_no_ln False \
        --prj_dropout 0.0 \
        --inf_latent_iterations 6 \
        --remove_eos True \
        --use_lora True"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $test_cmd"
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        eval "$test_cmd"
        
        if [[ $? -eq 0 ]]; then
            log_success "测试完成: ${MODEL_NAME}"
            log_info "结果目录: ${RESULT_DIR}"
        else
            log_error "测试失败: ${MODEL_NAME}"
            return 1
        fi
    fi
}

# =============================================================================
# 主函数
# =============================================================================
DRY_RUN="false"
FACTOR=""
ACTION=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        train|test|all|list)
            ACTION="$1"
            shift
            ;;
        -f|--factor)
            FACTOR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            ;;
    esac
done

# 检查参数
if [[ -z "$ACTION" ]]; then
    show_help
fi

if [[ "$ACTION" == "list" ]]; then
    echo "============================================================================"
    echo "  commonsense 数据集模型列表"
    echo "============================================================================"
    list_models
    exit 0
fi

if [[ -z "$FACTOR" ]]; then
    log_error "必须指定 -f/--factor 参数"
    echo ""
    echo "示例: $0 $ACTION -f 0.25"
    exit 1
fi

echo "============================================================================"
echo "  commonsense 数据集 simcon_sircl 模型"
echo "============================================================================"
echo "  操作: $ACTION"
echo "  Factor: $FACTOR"
echo "  结果目录: ${RESULT_DIR}"
echo "============================================================================"
echo ""

case "$ACTION" in
    train)
        run_training "$FACTOR"
        ;;
    test)
        run_testing "$FACTOR"
        ;;
    all)
        run_training "$FACTOR"
        run_testing "$FACTOR"
        ;;
esac

echo ""
log_success "完成!"
