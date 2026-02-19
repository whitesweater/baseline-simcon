#!/bin/bash
# =============================================================================
# commen_simcon_sircl 模型 trajectory_loss_factor 消融实验主脚本
# =============================================================================
# 实验配置:
#   - factor = 0.05 (新训练)
#   - factor = 0.1  (已完成，位于 /data/yhao/baseline/CODI/outputs/commen_simcon_sircl)
#   - factor = 0.2  (新训练)
#
# 用法:
#   ./run_commen_simcon_sircl_ablation.sh train      # 只训练
#   ./run_commen_simcon_sircl_ablation.sh test       # 只测试
#   ./run_commen_simcon_sircl_ablation.sh all        # 训练+测试
#   ./run_commen_simcon_sircl_ablation.sh --help     # 显示帮助
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# 加载环境配置
source "${PROJECT_DIR}/config.env" || { echo "Error: config.env not found"; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# 消融实验配置
ABLATION_NAME="trajectory_loss_factor"
FACTORS=("0.05" "0.1" "0.2")
RESULT_DIR="${CODI_RESULT_DIR}/ablation_${ABLATION_NAME}"
TRAINED_DIR="${CODI_SAVE_DIR}"

# 已训练模型映射
declare -A TRAINED_MODELS
TRAINED_MODELS["0.05"]="commen_simcon_sircl_factor005"
TRAINED_MODELS["0.1"]="commen_simcon_sircl"  # 已完成
TRAINED_MODELS["0.2"]="commen_simcon_sircl_factor02"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    echo "用法: $0 [train|test|all] [选项]"
    echo ""
    echo "命令:"
    echo "  train    只运行训练 (跳过 factor=0.1，因为已完成)"
    echo "  test     只运行测试"
    echo "  all      训练 + 测试"
    echo ""
    echo "选项:"
    echo "  --skip-existing  跳过已存在的训练模型"
    echo "  --dry-run        预览命令，不实际执行"
    echo "  -h, --help       显示帮助"
    echo ""
    echo "消融实验配置:"
    echo "  参数: trajectory_loss_factor"
    echo "  取值: 0.05, 0.1, 0.2"
    echo ""
    echo "模型位置:"
    for factor in "${FACTORS[@]}"; do
        model_name="${TRAINED_MODELS[$factor]}"
        model_path="${TRAINED_DIR}/${model_name}"
        if [[ -d "$model_path" ]]; then
            echo "  factor=$factor: $model_path ✓"
        else
            echo "  factor=$factor: $model_path (待训练)"
        fi
    done
    exit 0
}

# =============================================================================
# 训练函数
# =============================================================================
run_training() {
    log_info "开始消融实验训练..."
    log_info "参数: trajectory_loss_factor"
    log_info "取值: ${FACTORS[*]}"
    echo ""
    
    for factor in "${FACTORS[@]}"; do
        model_name="${TRAINED_MODELS[$factor]}"
        model_path="${TRAINED_DIR}/${model_name}"
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "factor = $factor"
        
        # 检查是否已训练
        if [[ -d "$model_path" ]]; then
            log_warn "模型已存在: $model_path"
            if [[ "$SKIP_EXISTING" == "true" ]]; then
                log_info "跳过训练..."
                continue
            fi
            log_info "由于模型已存在，跳过训练"
            continue
        fi
        
        # 选择对应的训练脚本
        case "$factor" in
            "0.05")
                script="${SCRIPT_DIR}/commen_simcon_sircl_factor005.sh"
                ;;
            "0.1")
                log_info "factor=0.1 已在 /data/yhao/baseline/CODI/outputs/commen_simcon_sircl 完成训练"
                continue
                ;;
            "0.2")
                script="${SCRIPT_DIR}/commen_simcon_sircl_factor02.sh"
                ;;
            *)
                log_error "未知的 factor 值: $factor"
                continue
                ;;
        esac
        
        if [[ ! -f "$script" ]]; then
            log_error "训练脚本不存在: $script"
            continue
        fi
        
        log_info "运行训练脚本: $script"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY-RUN] bash $script"
        else
            bash "$script"
            
            if [[ $? -eq 0 ]]; then
                log_success "factor=$factor 训练完成"
            else
                log_error "factor=$factor 训练失败"
            fi
        fi
    done
    
    echo ""
    log_success "所有训练任务完成!"
}

# =============================================================================
# 测试函数
# =============================================================================
run_testing() {
    log_info "开始消融实验测试..."
    mkdir -p "${RESULT_DIR}"
    
    # 收集所有可用的模型
    available_models=""
    for factor in "${FACTORS[@]}"; do
        model_name="${TRAINED_MODELS[$factor]}"
        
        # 特殊处理 0.1，它在 outputs 目录
        if [[ "$factor" == "0.1" ]]; then
            model_path="/data/yhao/baseline/CODI/outputs/commen_simcon_sircl"
        else
            model_path="${TRAINED_DIR}/${model_name}"
        fi
        
        if [[ -d "$model_path" ]]; then
            log_info "找到模型 factor=$factor: $model_path"
            available_models="${available_models} ${model_path}"
        else
            log_warn "模型不存在 factor=$factor: $model_path"
        fi
    done
    
    if [[ -z "$available_models" ]]; then
        log_error "没有找到可用的模型，请先运行训练"
        exit 1
    fi
    
    echo ""
    log_info "可用模型:$available_models"
    echo ""
    
    # 逐个测试模型
    for factor in "${FACTORS[@]}"; do
        model_name="${TRAINED_MODELS[$factor]}"
        
        # 特殊处理 0.1
        if [[ "$factor" == "0.1" ]]; then
            model_path="/data/yhao/baseline/CODI/outputs/commen_simcon_sircl"
        else
            model_path="${TRAINED_DIR}/${model_name}"
        fi
        
        if [[ ! -d "$model_path" ]]; then
            continue
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "测试模型: factor=$factor"
        log_info "Checkpoint: $model_path"
        
        test_cmd="python ${PROJECT_DIR}/test_multi_dataset.py \
            --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
            --ckpt_dir \"${model_path}\" \
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
            eval "$test_cmd"
        fi
    done
    
    echo ""
    log_success "所有测试完成!"
    log_info "结果目录: ${RESULT_DIR}"
    echo ""
    
    # 显示结果汇总
    if [[ -f "${RESULT_DIR}/summary/comparison_matrix.csv" ]]; then
        log_info "📊 结果对比矩阵:"
        column -t -s',' "${RESULT_DIR}/summary/comparison_matrix.csv" 2>/dev/null | head -20
    fi
    
    if [[ -f "${RESULT_DIR}/summary/all_results.csv" ]]; then
        log_info ""
        log_info "📋 详细结果:"
        cat "${RESULT_DIR}/summary/all_results.csv"
    fi
}

# =============================================================================
# 生成消融实验报告
# =============================================================================
generate_report() {
    log_info "生成消融实验报告..."
    
    report_file="${RESULT_DIR}/ablation_report.md"
    
    cat > "$report_file" << 'EOF'
# Trajectory Loss Factor 消融实验报告

## 实验配置

| 参数 | 取值 |
|------|------|
| trajectory_loss_factor | 0.05, 0.1, 0.2 |
| 数据集 | commonsense |
| 基础模型 | LLaMA 1B |

## 实验结果

EOF
    
    if [[ -f "${RESULT_DIR}/summary/all_results.csv" ]]; then
        echo "### 详细结果" >> "$report_file"
        echo '```' >> "$report_file"
        cat "${RESULT_DIR}/summary/all_results.csv" >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    log_success "报告已生成: $report_file"
}

# =============================================================================
# 主函数
# =============================================================================
SKIP_EXISTING="true"
DRY_RUN="false"

# 解析参数
ACTION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        train|test|all)
            ACTION="$1"
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING="true"
            shift
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

if [[ -z "$ACTION" ]]; then
    show_help
fi

echo "============================================================================"
echo "  Trajectory Loss Factor 消融实验"
echo "============================================================================"
echo "  参数取值: 0.05, 0.1, 0.2"
echo "  结果目录: ${RESULT_DIR}"
echo "  操作: $ACTION"
echo "============================================================================"
echo ""

case "$ACTION" in
    train)
        run_training
        ;;
    test)
        run_testing
        generate_report
        ;;
    all)
        run_training
        run_testing
        generate_report
        ;;
esac

echo ""
log_success "消融实验完成!"
echo ""
echo "查看结果:"
echo "  cat ${RESULT_DIR}/summary/all_results.csv"
echo "  cat ${RESULT_DIR}/ablation_report.md"
