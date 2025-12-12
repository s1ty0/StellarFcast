# generate_text_embeddings.py
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# on_debug = True

def _get_activity_level(stats):
    x_max = stats['x_max']
    x_med = stats['x_med']
    peak_ratio = stats['peak_ratio']
    duration = stats['duration']

    if peak_ratio > 100.0:
        return "无显著活动"
    if x_max < 0.1:
        return "无显著活动"
    if peak_ratio < 1.2 and x_max < 0.15:
        return "无显著活动"
    if peak_ratio < 2.5:
        return "微弱增强行为"
    if peak_ratio >= 2.5:
        if duration < 2:
            return "微弱增强行为"
        if peak_ratio >= 6.0 and x_max >= 0.5:
            return "强烈的爆发行为"
        else:
            return "明显的瞬态增强行为"
    return "微弱增强行为"

def _build_prompt(stats):
    activity = _get_activity_level(stats)
    peak_ratio = stats['peak_ratio']
    rise_rate_mean = stats['rise_rate'].mean()  # 注意：你代码中 rise_rate 是标量，不是数组！
    duration = stats['duration']
    return (
        f"在本次观测周期内，耀斑活动的全局统计特征如下："
        f"峰值通量相对于背景中位数的倍数为{peak_ratio:.2f}，"
        f"从起始到峰值的上升速率均值为{rise_rate_mean:.4f}单位/时间步，"
        f"爆发持续时间（半高全宽）为{duration:.1f}个时间步。"
        f"该序列表现出{activity}。"
    )

def compute_global_stats(lc_data):
    stats_list = []
    for x in lc_data:
        x_med = np.median(x)
        x_max = np.max(x)
        t_max = np.argmax(x)
        peak_ratio = x_max / (x_med + 1e-8)
        rise_rate = (x_max - x[0]) / max(t_max, 1)  # 标量！
        half_max = x_med + (x_max - x_med) * 0.5
        dur = np.sum(x > half_max) if np.any(x > half_max) else 0
        stats_list.append({
            'peak_ratio': peak_ratio,
            'rise_rate': rise_rate,      # 注意：是标量
            'duration': dur,
            'x_max': x_max,
            'x_med': x_med
        })
    return stats_list

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./myDataH")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--encoder", type=str, default="bert-chinese")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kepler", "tess"],
        required=True,
        help="Dataset identifier: 'kepler' or 'tess'"
    )
    args = parser.parse_args()
    if args.dataset == "kepler":
        args.data_root = "./myDataK"

    # 路径
    data_dir = os.path.join(args.data_root, args.split)
    lc_path = os.path.join(data_dir, "lc_data.npy")
    output_path = os.path.join(data_dir, f"text_embeddings_{args.encoder}.npy")

    # 加载光变曲线
    print(f"Loading {lc_path}...")
    lc_data = np.load(lc_path)  # (N, 512)
    # lc_data = lc_data[:10] # todo


    # 预计算统计量
    print("Computing global stats...")
    global_stats = compute_global_stats(lc_data)

    # 构建 prompts
    print("Building prompts...")
    prompts = [_build_prompt(stats) for stats in global_stats]
    # if on_debug:
    #     print(prompts)

    # 加载BERT
    encoder_path_map = {
        "minLM": "./textEncoder/all-MiniLM-L6-v2",
        "bert-chinese": "./textEncoder/bert-base-chinese",
    }

    model_path = encoder_path_map[args.encoder]
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 或 'cuda' 如果你有足够显存且想加速

    # 分批编码（防止 batch 太大）
    batch_size = 64
    all_embeddings = []

    print("Encoding prompts...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        # 移到 GPU（可选） TODO
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
        all_embeddings.append(embeddings)

    # 合并 & 保存
    final_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding shape: {final_embeddings.shape}")
    np.save(output_path, final_embeddings)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()