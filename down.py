from huggingface_hub import snapshot_download
import os
import os.path

# 稳下载配置
os.environ["HF_HUB_HTTP_TIMEOUT"] = "180"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# 在国内可启用镜像（可删）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 如需代理：
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"

# 不传 local_dir -> 落到 HF 默认缓存 ~/.cache/huggingface/hub

# 为了和其它模型目录保持一致：
# - 若 target_root 下已有 hub/，则用它作为 huggingface hub cache
# - 否则直接用 target_root 作为 cache_dir（会生成 models--* 目录）

local_dir = snapshot_download(
    repo_id="internlm/SIM_COT-LLaMA3-CODI-1B",
    repo_type="model",
    local_dir="/data/yhao/baseline/CODI",
    force_download=True,
)

print("snapshot 路径：", local_dir)