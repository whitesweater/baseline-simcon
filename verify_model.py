"""验证下载的模型是否正常"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 镜像配置（如果需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型路径（根据你的实际情况修改）
model_path = "/data/yhao/baseline/CODI/modelspace/models--internlm--SIM_COT-LLaMA3-CODI-1B/snapshots"

# 自动找到最新的 snapshot
import glob
snapshots = glob.glob(f"{model_path}/*")
if snapshots:
    model_path = max(snapshots, key=os.path.getmtime)
    print(f"使用模型路径：{model_path}")
else:
    print("未找到模型 snapshot，请检查路径")
    exit(1)

print("\n" + "="*60)
print("开始验证模型...")
print("="*60 + "\n")

try:
    # 1. 加载 tokenizer
    print("1️⃣  加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"   ✓ Tokenizer 加载成功")
    print(f"   - 词表大小: {len(tokenizer)}")
    
    # 2. 加载模型
    print("\n2️⃣  加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - 使用设备: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    print(f"   ✓ 模型加载成功")
    print(f"   - 模型类型: {model.config.model_type}")
    print(f"   - 隐藏层维度: {model.config.hidden_size}")
    print(f"   - 层数: {model.config.num_hidden_layers}")
    print(f"   - 注意力头数: {model.config.num_attention_heads}")
    
    # 3. 测试推理
    print("\n3️⃣  测试推理能力...")
    test_prompts = [
        "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\nA:",
        "The capital of France is",
        "1 + 1 =",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   测试 {i}: {prompt[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输出: {result[len(prompt):].strip()[:100]}")
    
    # 4. 检查特殊配置
    print("\n4️⃣  检查模型配置...")
    if hasattr(model.config, 'use_cache'):
        print(f"   - use_cache: {model.config.use_cache}")
    if hasattr(model.config, 'max_position_embeddings'):
        print(f"   - 最大序列长度: {model.config.max_position_embeddings}")
    
    # 5. 显存占用（GPU）
    if device == "cuda":
        print("\n5️⃣  GPU 显存占用...")
        print(f"   - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"   - 缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    print("\n" + "="*60)
    print("✅ 模型验证通过！模型工作正常。")
    print("="*60)
    
except Exception as e:
    print("\n" + "="*60)
    print("❌ 模型验证失败！")
    print("="*60)
    print(f"\n错误信息: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
