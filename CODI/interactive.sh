#!/bin/bash
# 交互式作业提交脚本
# 使用方法: bash interactive.sh

echo "正在申请交互式资源..."
echo "队列: emergency_gpu"
echo "CPU核心: 32"
echo "GPU: 1 张"
echo "内存: 128GB"
echo "时长: 24小时"
echo ""

# 提交交互式作业
srun -p emergency_gpu \
     -n 32 \
     --gres=gpu:1 \
     --mem=128G \
     --time=24:00:00 \
     --pty bash -c '
# 进入交互式环境后自动执行的命令

echo "=========================================="
echo "交互式作业已启动"
echo "主机: $(hostname)"
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "=========================================="
echo ""

# 激活虚拟环境
source /hpc2hdd/home/yhao481/jhupload/baseline/.venv/bin/activate

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false

# 显示环境信息
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA可用: {torch.cuda.is_available()}\"); print(f\"GPU数量: {torch.cuda.device_count()}\")"
echo ""
echo "=========================================="
echo "环境已配置完成，现在可以开始工作"
echo "提示: 使用 Ctrl+D 或输入 exit 退出"
echo "=========================================="
echo ""

# 进入交互式 bash
exec bash
'
