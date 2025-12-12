import argparse
import os
import numpy as np


def remove_invalid_flares(lc_data, label_data, mask_data, min_rise=0.001):
    """
    删除 label=1 且 max_rise <= min_rise 的样本（彻底移除）

    Returns:
        cleaned_lc: np.ndarray, shape [N_clean, 1, 512]
        cleaned_labels: np.ndarray, shape [N_clean,]
        removed_indices: list of indices that were removed
    """
    keep_mask = np.ones(len(label_data), dtype=bool)
    removed_indices = []
    valid_rises = []

    for i in range(len(label_data)):
        if label_data[i] == 1:
            lc = lc_data[i, 0, :]  # shape [512]
            diff = lc[1:] - lc[:-1]
            max_rise = np.max(diff)

            if max_rise <= min_rise:
                keep_mask[i] = False
                removed_indices.append(i)
            else:
                valid_rises.append(max_rise)

    cleaned_lc = lc_data[keep_mask]
    cleaned_labels = label_data[keep_mask]
    cleaned_mask = mask_data[keep_mask]

    print(f"原始样本数: {len(label_data)}")
    print(f"原始耀斑数: {np.sum(label_data == 1)}")
    print(f"删除样本数: {len(removed_indices)} (全部为 label=1 且 max_rise <= {min_rise})")
    print(f"清洗后样本数: {len(cleaned_labels)}")
    print(f"清洗后耀斑数: {np.sum(cleaned_labels == 1)}")
    print(f"清洗后对应的历史耀斑记录数: {len(cleaned_mask)}")


    # 计算 q5（仅用于 train 集）
    if valid_rises:
        q5 = np.percentile(valid_rises, 5)
        recommended = q5 * 0.8
        print(f"q5: {q5:.6f} → 推荐 rise_threshold: {recommended:.6f}")
    else:
        recommended = 0.005

    return cleaned_lc, cleaned_labels,cleaned_mask, recommended


def main():
    parser = argparse.ArgumentParser(description="Process dataset with robust patching and splitting.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kepler", "tess"],
        required=True,
        help="Dataset identifier: 'kepler' or 'tess'"
    )

    args = parser.parse_args()
    if args.dataset == "kepler":
        output_root = "../myDataK"  # 新目录名，避免混淆
    elif args.dataset == "tess":
        output_root = "../myDataH"  # 新目录名，避免混淆
    else:
        raise ValueError("Unsupported dataset. Choose 'kepler' or 'tess'.")

    data_root = "../no_leak_dataset"
    splits = ["train", "val", "test"]
    min_rise = 0.001

    os.makedirs(output_root, exist_ok=True)
    global_threshold = None

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"🧹 严格清洗 {split} 集（删除无效样本）")
        print(f"{'=' * 50}")

        lc_path = os.path.join(data_root, split, "lc_data.npy")
        label_path = os.path.join(data_root, split, "label_data.npy")
        mask_path = os.path.join(data_root, split, "mask_data.npy")

        lc_data = np.load(lc_path)
        label_data = np.load(label_path)
        mask_data = np.load(mask_path)

        print(f"加载: {lc_data.shape}, {label_data.shape}, {mask_data.shape}")

        cleaned_lc, cleaned_labels, cleaned_mask, rec_thresh = remove_invalid_flares(
            lc_data, label_data,mask_data, min_rise=min_rise
        )

        if split == "train":
            global_threshold = rec_thresh

        # 保存清洗后的数据
        out_dir = os.path.join(output_root, split)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "lc_data.npy"), cleaned_lc)
        np.save(os.path.join(out_dir, "label_data.npy"), cleaned_labels)
        np.save(os.path.join(out_dir, "mask_data.npy"), cleaned_mask)


        print(f"✅ 已保存至: {out_dir}")
        print("After clean, our data shape is ")
        print(f"lc_data.shape: {cleaned_lc.shape}")
        print(f"label_data.shape: {cleaned_labels.shape}")
        print(f"mask_data.shape: {cleaned_mask.shape}")


    print(f"\n{'=' * 60}")
    print(f"🎯 最终推荐 rise_threshold（基于清洗后的 train 集）: {global_threshold:.6f}")
    print(f"使用建议: PhysicsRegularizedLoss(rise_threshold={global_threshold:.6f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()