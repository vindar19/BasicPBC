# -*- coding: utf-8 -*-
"""
BasicPBC 多帧人工上色结果对比工具

用途：
    比较 PaintMan 人工上色 TGA
    与 BasicPBC 输出 PNG

默认目录结构：

CUT_03/
├── color_reference/
│   ├── c0002.tga
│   ├── c0003.tga
│   ├── c0004.tga
│   └── c0005.tga
│
└── basicpbc_input/
    └── basicpbc_input_keepline/
        ├── 0000.png
        ├── 0001.png
        ├── 0002.png
        └── 0003.png

比较时：
    0000 <-> c0002
    0001 <-> c0003
    0002 <-> c0004
    0003 <-> c0005

特别处理：
    BasicPBC 使用 --keep_line 后会额外保留黑色线稿。
    因此同时计算：

    1. ALL PIXELS
       所有像素直接比较

    2. COLOR PIXELS
       排除黑色线稿后的颜色比较

    3. WHITE PIXELS
       统计纯白像素变化

    4. DIFFERENCE BBOX
       统计差异区域

    5. 每帧结果 + 总体趋势

运行：

python tools\compare_basicpbc_frames.py

也可以指定目录：

python tools\compare_basicpbc_frames.py ^
    --cut "G:\paint_assistant\ref_prj\DATA\CUT_03"

"""

import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# 配置
# ============================================================

DEFAULT_CUT = r"G:\paint_assistant\ref_prj\DATA\CUT_03"

DEFAULT_REFERENCE = "color_reference"

DEFAULT_RESULT = os.path.join(
    "basicpbc_input",
    "basicpbc_input_keepline"
)

DEFAULT_OUTPUT = os.path.join(
    "basicpbc_input",
    "compare_report"
)


# ============================================================
# 图片读取
# ============================================================

def load_rgb(path):
    """
    读取图片并转换为 RGB。
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(str(path))

    im = Image.open(path)

    rgb = im.convert("RGB")

    arr = np.asarray(rgb, dtype=np.uint8)

    return arr


# ============================================================
# 基础统计
# ============================================================

def basic_stats(a, b):
    """
    对两个 RGB 图像进行完整像素比较。
    """

    if a.shape != b.shape:
        raise ValueError(
            f"尺寸不一致：A={a.shape}, B={b.shape}"
        )

    af = a.astype(np.int16)
    bf = b.astype(np.int16)

    diff = np.abs(af - bf)

    pixel_diff = np.any(diff != 0, axis=2)

    total_pixels = a.shape[0] * a.shape[1]

    different_pixels = int(pixel_diff.sum())

    different_percent = (
        different_pixels / total_pixels * 100.0
    )

    max_diff = int(diff.max())

    mean_diff = float(diff.mean())

    mae_per_pixel = diff.mean(axis=2)

    mae = float(mae_per_pixel.mean())

    mse = float((diff.astype(np.float32) ** 2).mean())

    rmse = float(np.sqrt(mse))

    return {
        "total_pixels": total_pixels,
        "different_pixels": different_pixels,
        "different_percent": different_percent,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mae": mae,
        "rmse": rmse,
        "diff": diff,
        "pixel_diff": pixel_diff,
    }


# ============================================================
# 颜色区域统计
# ============================================================

def color_only_stats(a, b):
    """
    排除黑色线稿后比较颜色。

    这里不简单排除：
        A == black
        或
        B == black

    而是构造一个更合理的“非黑色区域”：

        A 不是纯黑
        且
        B 不是纯黑

    这样可以避免 --keep_line 导致黑线差异污染结果。
    """

    af = a.astype(np.int16)
    bf = b.astype(np.int16)

    black_a = np.all(a == [0, 0, 0], axis=2)
    black_b = np.all(b == [0, 0, 0], axis=2)

    valid = ~(black_a | black_b)

    count = int(valid.sum())

    if count == 0:
        return {
            "valid_pixels": 0,
            "different_pixels": 0,
            "different_percent": 0.0,
            "max_diff": 0,
            "mean_diff": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
        }

    diff = np.abs(af - bf)

    valid_diff = diff[valid]

    pixel_diff = np.any(valid_diff != 0, axis=1)

    different_pixels = int(pixel_diff.sum())

    different_percent = (
        different_pixels / count * 100.0
    )

    max_diff = int(valid_diff.max())

    mean_diff = float(valid_diff.mean())

    mse = float(
        (valid_diff.astype(np.float32) ** 2).mean()
    )

    rmse = float(np.sqrt(mse))

    return {
        "valid_pixels": count,
        "different_pixels": different_pixels,
        "different_percent": different_percent,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mae": mean_diff,
        "rmse": rmse,
    }


# ============================================================
# 黑色线稿统计
# ============================================================

def black_stats(a, b):

    black_a = np.all(a == [0, 0, 0], axis=2)

    black_b = np.all(b == [0, 0, 0], axis=2)

    count_a = int(black_a.sum())

    count_b = int(black_b.sum())

    same = np.logical_and(
        black_a,
        black_b
    )

    only_a = np.logical_and(
        black_a,
        ~black_b
    )

    only_b = np.logical_and(
        ~black_a,
        black_b
    )

    return {
        "black_a": count_a,
        "black_b": count_b,
        "black_same": int(same.sum()),
        "black_only_a": int(only_a.sum()),
        "black_only_b": int(only_b.sum()),
    }


# ============================================================
# 白色统计
# ============================================================

def white_stats(a, b):

    white_a = np.all(a == [255, 255, 255], axis=2)

    white_b = np.all(b == [255, 255, 255], axis=2)

    count_a = int(white_a.sum())

    count_b = int(white_b.sum())

    white_to_nonwhite = np.logical_and(
        white_a,
        ~white_b
    )

    nonwhite_to_white = np.logical_and(
        ~white_a,
        white_b
    )

    white_same = np.logical_and(
        white_a,
        white_b
    )

    return {
        "white_a": count_a,
        "white_b": count_b,
        "white_same": int(white_same.sum()),
        "white_to_nonwhite": int(
            white_to_nonwhite.sum()
        ),
        "nonwhite_to_white": int(
            nonwhite_to_white.sum()
        ),
    }


# ============================================================
# 差异区域
# ============================================================

def difference_bbox(a, b):

    diff = np.any(a != b, axis=2)

    ys, xs = np.where(diff)

    if len(xs) == 0:
        return None

    return {
        "xmin": int(xs.min()),
        "xmax": int(xs.max()),
        "ymin": int(ys.min()),
        "ymax": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


# ============================================================
# 差异图
# ============================================================

def save_diff_image(a, b, output_path):

    af = a.astype(np.int16)
    bf = b.astype(np.int16)

    diff = np.abs(af - bf)

    # 三通道差异直接显示
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    Image.fromarray(diff, mode="RGB").save(output_path)


# ============================================================
# 彩色差异图
# ============================================================

def save_color_diff_image(a, b, output_path):

    af = a.astype(np.int16)
    bf = b.astype(np.int16)

    diff = np.abs(af - bf)

    magnitude = diff.max(axis=2)

    # 将差异放大，方便肉眼观察
    magnitude = np.clip(
        magnitude.astype(np.float32) * 3.0,
        0,
        255
    ).astype(np.uint8)

    out = np.stack(
        [magnitude, magnitude, magnitude],
        axis=2
    )

    Image.fromarray(out, mode="RGB").save(output_path)


# ============================================================
# 单帧比较
# ============================================================

def compare_frame(
    frame_index,
    reference_path,
    result_path,
    output_dir
):

    print()
    print("=" * 70)
    print(f"FRAME {frame_index:04d}")
    print("=" * 70)

    print()
    print("REFERENCE:")
    print(reference_path)

    print()
    print("BASICPBC:")
    print(result_path)

    reference = load_rgb(reference_path)

    result = load_rgb(result_path)

    print()
    print("Shape:")
    print("  reference =", reference.shape)
    print("  basicpbc   =", result.shape)

    if reference.shape != result.shape:
        raise ValueError(
            f"图片尺寸不一致: "
            f"{reference.shape} vs {result.shape}"
        )

    # --------------------------------------------------------
    # 全像素
    # --------------------------------------------------------

    all_stats = basic_stats(
        reference,
        result
    )

    # --------------------------------------------------------
    # 排除黑线
    # --------------------------------------------------------

    color_stats = color_only_stats(
        reference,
        result
    )

    # --------------------------------------------------------
    # 黑线
    # --------------------------------------------------------

    black = black_stats(
        reference,
        result
    )

    # --------------------------------------------------------
    # 白色
    # --------------------------------------------------------

    white = white_stats(
        reference,
        result
    )

    # --------------------------------------------------------
    # bbox
    # --------------------------------------------------------

    bbox = difference_bbox(
        reference,
        result
    )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print()
    print("【ALL PIXELS】")

    print(
        f"  different pixels : "
        f"{all_stats['different_pixels']:,}"
    )

    print(
        f"  different %      : "
        f"{all_stats['different_percent']:.6f}%"
    )

    print(
        f"  MAX RGB DIFF     : "
        f"{all_stats['max_diff']}"
    )

    print(
        f"  MEAN RGB DIFF    : "
        f"{all_stats['mean_diff']:.6f}"
    )

    print(
        f"  MAE              : "
        f"{all_stats['mae']:.6f}"
    )

    print(
        f"  RMSE             : "
        f"{all_stats['rmse']:.6f}"
    )

    print()
    print("【COLOR ONLY / BLACK LINE REMOVED】")

    print(
        f"  valid pixels     : "
        f"{color_stats['valid_pixels']:,}"
    )

    print(
        f"  different pixels : "
        f"{color_stats['different_pixels']:,}"
    )

    print(
        f"  different %      : "
        f"{color_stats['different_percent']:.6f}%"
    )

    print(
        f"  MAX RGB DIFF     : "
        f"{color_stats['max_diff']}"
    )

    print(
        f"  MEAN RGB DIFF    : "
        f"{color_stats['mean_diff']:.6f}"
    )

    print(
        f"  RMSE             : "
        f"{color_stats['rmse']:.6f}"
    )

    print()
    print("【BLACK LINE】")

    print(
        f"  reference black  : "
        f"{black['black_a']:,}"
    )

    print(
        f"  basicpbc black   : "
        f"{black['black_b']:,}"
    )

    print(
        f"  same black       : "
        f"{black['black_same']:,}"
    )

    print(
        f"  only reference   : "
        f"{black['black_only_a']:,}"
    )

    print(
        f"  only basicpbc    : "
        f"{black['black_only_b']:,}"
    )

    print()
    print("【PURE WHITE】")

    print(
        f"  reference white  : "
        f"{white['white_a']:,}"
    )

    print(
        f"  basicpbc white   : "
        f"{white['white_b']:,}"
    )

    print(
        f"  same white       : "
        f"{white['white_same']:,}"
    )

    print(
        f"  white -> color   : "
        f"{white['white_to_nonwhite']:,}"
    )

    print(
        f"  color -> white   : "
        f"{white['nonwhite_to_white']:,}"
    )

    print()
    print("【DIFFERENCE BBOX】")

    if bbox is None:

        print("  No difference.")

    else:

        print(
            f"  X: {bbox['xmin']} ~ "
            f"{bbox['xmax']}"
        )

        print(
            f"  Y: {bbox['ymin']} ~ "
            f"{bbox['ymax']}"
        )

        print(
            f"  size: "
            f"{bbox['width']} x "
            f"{bbox['height']}"
        )

    # --------------------------------------------------------
    # 保存差异图
    # --------------------------------------------------------

    frame_name = f"{frame_index:04d}"

    raw_diff_path = (
        output_dir /
        f"{frame_name}_diff.png"
    )

    color_diff_path = (
        output_dir /
        f"{frame_name}_color_diff.png"
    )

    save_diff_image(
        reference,
        result,
        raw_diff_path
    )

    save_color_diff_image(
        reference,
        result,
        color_diff_path
    )

    print()
    print("差异图：")

    print(
        f"  {raw_diff_path}"
    )

    print(
        f"  {color_diff_path}"
    )

    return {
        "frame": frame_index,

        "reference": str(reference_path),
        "result": str(result_path),

        "all_different_pixels":
            all_stats["different_pixels"],

        "all_different_percent":
            all_stats["different_percent"],

        "all_mae":
            all_stats["mae"],

        "all_rmse":
            all_stats["rmse"],

        "all_max_diff":
            all_stats["max_diff"],

        "color_valid_pixels":
            color_stats["valid_pixels"],

        "color_different_pixels":
            color_stats["different_pixels"],

        "color_different_percent":
            color_stats["different_percent"],

        "color_mae":
            color_stats["mae"],

        "color_rmse":
            color_stats["rmse"],

        "color_max_diff":
            color_stats["max_diff"],

        "reference_black":
            black["black_a"],

        "basicpbc_black":
            black["black_b"],

        "reference_white":
            white["white_a"],

        "basicpbc_white":
            white["white_b"],

        "white_to_nonwhite":
            white["white_to_nonwhite"],

        "nonwhite_to_white":
            white["nonwhite_to_white"],

        "bbox_xmin":
            "" if bbox is None else bbox["xmin"],

        "bbox_xmax":
            "" if bbox is None else bbox["xmax"],

        "bbox_ymin":
            "" if bbox is None else bbox["ymin"],

        "bbox_ymax":
            "" if bbox is None else bbox["ymax"],
    }


# ============================================================
# 自动寻找对应帧
# ============================================================

def find_reference_frames(reference_dir):

    files = []

    for p in Path(reference_dir).glob("*.tga"):

        name = p.stem.lower()

        if name.startswith("c"):

            number = name[1:]

            if number.isdigit():

                files.append(
                    (int(number), p)
                )

    files.sort(
        key=lambda x: x[0]
    )

    return files


def find_result_frames(result_dir):

    files = []

    for p in Path(result_dir).glob("*.png"):

        name = p.stem

        if name.isdigit():

            files.append(
                (int(name), p)
            )

    files.sort(
        key=lambda x: x[0]
    )

    return files


# ============================================================
# 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="BasicPBC 多帧人工上色结果比较"
    )

    parser.add_argument(
        "--cut",
        default=DEFAULT_CUT,
        help="CUT 根目录"
    )

    parser.add_argument(
        "--reference",
        default=None,
        help="人工上色参考目录"
    )

    parser.add_argument(
        "--result",
        default=None,
        help="BasicPBC 输出目录"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="分析结果目录"
    )

    args = parser.parse_args()

    cut_dir = Path(args.cut)

    reference_dir = (
        Path(args.reference)
        if args.reference
        else cut_dir / DEFAULT_REFERENCE
    )

    result_dir = (
        Path(args.result)
        if args.result
        else cut_dir / DEFAULT_RESULT
    )

    output_dir = (
        Path(args.output)
        if args.output
        else cut_dir / DEFAULT_OUTPUT
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("BasicPBC 多帧人工上色结果分析")
    print("=" * 70)

    print()
    print("CUT:")
    print(cut_dir)

    print()
    print("REFERENCE:")
    print(reference_dir)

    print()
    print("BASICPBC:")
    print(result_dir)

    print()
    print("OUTPUT:")
    print(output_dir)

    # --------------------------------------------------------
    # 找文件
    # --------------------------------------------------------

    reference_files = find_reference_frames(
        reference_dir
    )

    result_files = find_result_frames(
        result_dir
    )

    print()
    print(
        f"发现人工参考："
        f"{len(reference_files)} 张"
    )

    for number, path in reference_files:

        print(
            f"  c{number:04d}.tga"
        )

    print()
    print(
        f"发现 BasicPBC："
        f"{len(result_files)} 张"
    )

    for number, path in result_files:

        print(
            f"  {number:04d}.png"
        )

    # --------------------------------------------------------
    # 建立映射
    #
    # 第一个人工参考帧 c0002
    # 对应 BasicPBC 0000
    #
    # c0003 -> 0001
    # c0004 -> 0002
    # c0005 -> 0003
    # --------------------------------------------------------

    if not reference_files:
        raise RuntimeError(
            "没有找到 color_reference/*.tga"
        )

    if not result_files:
        raise RuntimeError(
            "没有找到 BasicPBC PNG"
        )

    reference_files.sort(
        key=lambda x: x[0]
    )

    result_files.sort(
        key=lambda x: x[0]
    )

    count = min(
        len(reference_files),
        len(result_files)
    )

    if len(reference_files) != len(result_files):

        print()
        print(
            "[WARNING] "
            "人工参考帧数量和 BasicPBC "
            "结果帧数量不同。"
        )

        print(
            "将按照排序后的前 "
            f"{count} 帧进行比较。"
        )

    # --------------------------------------------------------
    # 开始比较
    # --------------------------------------------------------

    results = []

    for i in range(count):

        reference_number, reference_path = (
            reference_files[i]
        )

        result_number, result_path = (
            result_files[i]
        )

        print()
        print(
            f"\n映射："
            f"c{reference_number:04d}.tga"
            f"  <->  "
            f"{result_number:04d}.png"
        )

        try:

            result = compare_frame(
                i,
                reference_path,
                result_path,
                output_dir
            )

            result["reference_frame"] = (
                reference_number
            )

            result["result_frame"] = (
                result_number
            )

            results.append(result)

        except Exception as e:

            print()
            print(
                f"[ERROR] FRAME {i:04d}:"
            )

            print(e)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir /
        "comparison_summary.csv"
    )

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(results)

    # --------------------------------------------------------
    # 总结
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("总体结果")
    print("=" * 70)

    print()

    print(
        f"{'FRAME':<8}"
        f"{'ALL DIFF%':>14}"
        f"{'COLOR DIFF%':>16}"
        f"{'COLOR MAE':>14}"
        f"{'COLOR RMSE':>14}"
    )

    print("-" * 70)

    for r in results:

        print(
            f"{r['frame']:04d}    "
            f"{r['all_different_percent']:>12.4f}%"
            f"{r['color_different_percent']:>14.4f}%"
            f"{r['color_mae']:>14.4f}"
            f"{r['color_rmse']:>14.4f}"
        )

    # --------------------------------------------------------
    # 平均值
    # --------------------------------------------------------

    if results:

        avg_all = np.mean([
            r["all_different_percent"]
            for r in results
        ])

        avg_color = np.mean([
            r["color_different_percent"]
            for r in results
        ])

        avg_mae = np.mean([
            r["color_mae"]
            for r in results
        ])

        avg_rmse = np.mean([
            r["color_rmse"]
            for r in results
        ])

        print("-" * 70)

        print(
            f"{'AVERAGE':<8}"
            f"{avg_all:>12.4f}%"
            f"{avg_color:>14.4f}%"
            f"{avg_mae:>14.4f}"
            f"{avg_rmse:>14.4f}"
        )

    print()
    print("=" * 70)

    print()
    print(
        "CSV："
    )

    print(
        csv_path
    )

    print()
    print(
        "差异图目录："
    )

    print(
        output_dir
    )

    print()
    print("分析完成。")


if __name__ == "__main__":
    main()