# -*- coding: utf-8 -*-

"""
BasicPBC / PaintMan TGA 数据准备

============================================================
目录结构
============================================================

CUT_xx
│
├─ tga
│   ├─ c0002.tga
│   ├─ c0003.tga
│   ├─ c0004.tga
│   └─ ...
│
└─ color_reference
    └─ c0002.tga


输出：

basicpbc_input
│
├─ gt
│   └─ 0000.png
│
├─ line
│   ├─ 0000.png
│   ├─ 0001.png
│   ├─ ...
│
└─ frame_map.json


============================================================
PaintMan 规则
============================================================

LINE：

    RGB == (255,255,255)
        -> Alpha = 0

    其它 RGB
        -> Alpha = 255


注意：

    254,255,255
    255,254,255
    255,255,254
    250,250,250

都不是透明。

GT：

    TGA RGB
        ↓
    RGB PNG

不做任何颜色修改。

不做：

    近白色处理
    Alpha 处理
    颜色量化
    颜色校正
    定位孔删除
    线稿删除


============================================================
重要
============================================================

支持命令：

    python prepare_cut.py

默认：

    DATA/CUT_03


也支持：

    python prepare_cut.py --cut-dir "G:/.../DATA/CUT_02"

此时只处理 CUT_02。

GUI 会使用 --cut-dir，
因此不会再出现：

    GUI 显示 CUT_02
    实际却处理 CUT_03

的问题。
"""

import argparse
from pathlib import Path
import json
import re
import shutil
import sys

import numpy as np
from PIL import Image


# ============================================================
# 基础路径
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

REF_PRJ_DIR = PROJECT_DIR.parent

DATA_DIR = REF_PRJ_DIR / "DATA"


# ============================================================
# 默认 CUT
# ============================================================

DEFAULT_CUT_DIR = DATA_DIR / "CUT_03"


# ============================================================
# PaintMan 透明规则
# ============================================================

TRANSPARENT_RGB = (
    255,
    255,
    255
)


# ============================================================
# 是否清理旧输出
# ============================================================

CLEAN_OUTPUT = True


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BasicPBC PaintMan TGA 数据准备"
    )

    parser.add_argument(
        "--cut-dir",
        type=str,
        default=None,
        help="指定要处理的 CUT 文件夹"
    )

    return parser.parse_args()


# ============================================================
# 获取 CUT
# ============================================================

def resolve_cut_dir(args):

    if args.cut_dir:

        cut_dir = Path(
            args.cut_dir
        ).expanduser().resolve()

    else:

        cut_dir = (
            DEFAULT_CUT_DIR
            .resolve()
        )

    return cut_dir


# ============================================================
# 自然排序
# ============================================================

def natural_sort(files):

    def key(path):

        match = re.search(
            r"(\d+)",
            path.name
        )

        if match:

            return (
                int(match.group(1)),
                path.name.lower()
            )

        return (
            999999999,
            path.name.lower()
        )

    return sorted(
        files,
        key=key
    )


# ============================================================
# 获取 TGA
# ============================================================

def get_tga_files(directory):

    if not directory.exists():
        return []

    files = []

    for p in directory.iterdir():

        if not p.is_file():
            continue

        if p.suffix.lower() == ".tga":

            files.append(p)

    return natural_sort(
        files
    )


# ============================================================
# 清理目录
# ============================================================

def clean_directory(directory):

    if not directory.exists():
        return

    for p in directory.iterdir():

        if p.is_file() or p.is_symlink():

            p.unlink()

        elif p.is_dir():

            shutil.rmtree(p)


# ============================================================
# 获取文件名中的帧号
# ============================================================

def get_frame_number(filename):

    match = re.search(
        r"(\d+)",
        filename
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# LINE TGA -> RGBA PNG
# ============================================================

def convert_line_tga(
    src_path,
    dst_path
):

    """
    PaintMan LINE TGA → RGBA PNG

    只有：

        RGB == (255,255,255)

    才透明。

    其它所有颜色都保留。
    """

    # --------------------------------------------------------
    # 打开
    # --------------------------------------------------------

    im = Image.open(
        src_path
    )

    original_mode = im.mode

    # --------------------------------------------------------
    # 转 RGB
    # --------------------------------------------------------

    rgb_im = im.convert(
        "RGB"
    )

    rgb = np.array(
        rgb_im,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 严格纯白判断
    # --------------------------------------------------------

    white_mask = np.all(
        rgb == np.array(
            TRANSPARENT_RGB,
            dtype=np.uint8
        ),
        axis=2
    )

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------

    alpha = np.where(
        white_mask,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # RGBA
    # --------------------------------------------------------

    rgba = np.dstack(
        [
            rgb,
            alpha
        ]
    )

    out = Image.fromarray(
        rgba,
        mode="RGBA"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    out.save(
        dst_path,
        format="PNG"
    )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    total = white_mask.size

    transparent = int(
        white_mask.sum()
    )

    line = int(
        (~white_mask).sum()
    )

    return {

        "width":
            out.width,

        "height":
            out.height,

        "mode":
            original_mode,

        "transparent":
            transparent,

        "line":
            line,

        "total":
            total
    }


# ============================================================
# GT TGA -> RGB PNG
# ============================================================

def convert_gt_tga(
    src_path,
    dst_path
):

    """
    PaintMan GT TGA → RGB PNG

    不做颜色修改。

    TGA RGB
        ↓
    RGB PNG
    """

    # --------------------------------------------------------
    # 打开
    # --------------------------------------------------------

    im = Image.open(
        src_path
    )

    original_mode = im.mode

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_im = im.convert(
        "RGB"
    )

    rgb_before = np.array(
        rgb_im,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    rgb_im.save(
        dst_path,
        format="PNG"
    )

    # --------------------------------------------------------
    # 再读取验证
    # --------------------------------------------------------

    rgb_after = np.array(
        Image.open(
            dst_path
        ).convert("RGB"),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 比较
    # --------------------------------------------------------

    diff = np.abs(
        rgb_before.astype(
            np.int16
        )
        -
        rgb_after.astype(
            np.int16
        )
    )

    max_diff = int(
        diff.max()
    )

    mean_diff = float(
        diff.mean()
    )

    different_pixels = int(
        np.any(
            diff != 0,
            axis=2
        ).sum()
    )

    return {

        "width":
            rgb_im.width,

        "height":
            rgb_im.height,

        "source_mode":
            original_mode,

        "output_mode":
            rgb_im.mode,

        "max_diff":
            max_diff,

        "mean_diff":
            mean_diff,

        "different_pixels":
            different_pixels
    }


# ============================================================
# 主程序
# ============================================================

def main():

    args = parse_args()

    CUT_DIR = resolve_cut_dir(
        args
    )

    TGA_DIR = CUT_DIR / "tga"

    REFERENCE_DIR = (
        CUT_DIR /
        "color_reference"
    )

    OUTPUT_DIR = (
        CUT_DIR /
        "basicpbc_input"
    )

    GT_DIR = (
        OUTPUT_DIR /
        "gt"
    )

    LINE_DIR = (
        OUTPUT_DIR /
        "line"
    )

    # ========================================================
    # 标题
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "BasicPBC TGA 数据准备"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"CUT_DIR          : {CUT_DIR}"
    )

    print(
        f"TGA_DIR          : {TGA_DIR}"
    )

    print(
        f"REFERENCE_DIR    : {REFERENCE_DIR}"
    )

    print(
        f"OUTPUT_DIR       : {OUTPUT_DIR}"
    )

    print(
        "LINE TRANSPARENT : RGB == (255,255,255)"
    )

    # ========================================================
    # 检查 CUT
    # ========================================================

    if not CUT_DIR.exists():

        print()

        print(
            "[ERROR] CUT 文件夹不存在："
        )

        print(
            CUT_DIR
        )

        sys.exit(1)

    if not TGA_DIR.exists():

        print()

        print(
            "[ERROR] 找不到 tga 目录："
        )

        print(
            TGA_DIR
        )

        sys.exit(1)

    if not REFERENCE_DIR.exists():

        print()

        print(
            "[ERROR] 找不到 color_reference 目录："
        )

        print(
            REFERENCE_DIR
        )

        sys.exit(1)

    # ========================================================
    # 查找文件
    # ========================================================

    line_files = get_tga_files(
        TGA_DIR
    )

    reference_files = get_tga_files(
        REFERENCE_DIR
    )

    print()

    print(
        f"发现线稿 TGA: {len(line_files)} 张"
    )

    for p in line_files:

        print(
            f"  {p.name}"
        )

    print()

    print(
        f"发现色见本 TGA: {len(reference_files)} 张"
    )

    for p in reference_files:

        print(
            f"  {p.name}"
        )

    # ========================================================
    # 检查数量
    # ========================================================

    if len(line_files) == 0:

        print()

        print(
            "[ERROR] tga 文件夹为空。"
        )

        sys.exit(1)

    if len(reference_files) == 0:

        print()

        print(
            "[ERROR] color_reference 文件夹为空。"
        )

        sys.exit(1)

    # ========================================================
    # 创建输出
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 清理旧数据
    # ========================================================

    if CLEAN_OUTPUT:

        print()

        print(
            "清理旧 gt / line ..."
        )

        clean_directory(
            GT_DIR
        )

        clean_directory(
            LINE_DIR
        )

        frame_map_path = (
            OUTPUT_DIR /
            "frame_map.json"
        )

        if frame_map_path.exists():

            frame_map_path.unlink()

    # ========================================================
    # LINE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "转换线稿"
    )

    print(
        "=" * 70
    )

    line_frame_map = []

    for index, src in enumerate(
        line_files
    ):

        dst = (
            LINE_DIR /
            f"{index:04d}.png"
        )

        info = convert_line_tga(
            src,
            dst
        )

        line_frame_map.append(
            {
                "index":
                    index,

                "basicpbc_frame":
                    f"{index:04d}",

                "source_file":
                    src.name,

                "source_frame":
                    get_frame_number(
                        src.name
                    ),

                "output_file":
                    dst.name
            }
        )

        print()

        print(
            f"[LINE {index:04d}] "
            f"{src.name} -> "
            f"{dst.name}"
        )

        print(
            f"  "
            f"{info['width']}x"
            f"{info['height']} "
            f"{info['mode']}"
        )

        print(
            f"  "
            f"white/transparent="
            f"{info['transparent']:,} "
            f"line="
            f"{info['line']:,}"
        )

    # ========================================================
    # GT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "转换色见本"
    )

    print(
        "=" * 70
    )

    # 当前 BasicPBC 工作流：
    #
    # 使用 color_reference 中第一张
    #
    reference_file = reference_files[0]

    gt_path = (
        GT_DIR /
        "0000.png"
    )

    info = convert_gt_tga(
        reference_file,
        gt_path
    )

    print()

    print(
        f"[GT] "
        f"{reference_file.name} -> "
        f"0000.png"
    )

    print(
        f"  "
        f"{info['width']}x"
        f"{info['height']}"
    )

    print(
        f"  source mode : "
        f"{info['source_mode']}"
    )

    print(
        f"  output mode : "
        f"{info['output_mode']}"
    )

    print(
        f"  RGB max diff: "
        f"{info['max_diff']}"
    )

    print(
        f"  RGB mean diff: "
        f"{info['mean_diff']}"
    )

    print(
        f"  different pixels: "
        f"{info['different_pixels']}"
    )

    # ========================================================
    # GT 严格验证
    # ========================================================

    if info["max_diff"] != 0:

        print()

        print(
            "[ERROR] "
            "TGA → PNG RGB 发生变化！"
        )

        sys.exit(1)

    print()

    print(
        "[OK] GT RGB 完全保持。"
    )

    # ========================================================
    # frame_map
    # ========================================================

    frame_map = {

        "cut":
            CUT_DIR.name,

        "cut_dir":
            str(CUT_DIR),

        "line_count":
            len(line_files),

        "gt":
            {
                "source_file":
                    reference_file.name,

                "output_file":
                    "gt/0000.png"
            },

        "lines":
            line_frame_map
    }

    frame_map_path = (
        OUTPUT_DIR /
        "frame_map.json"
    )

    with open(
        frame_map_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            frame_map,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # 完成
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "完成"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "BasicPBC 输入目录："
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "GT："
    )

    print(
        gt_path
    )

    print()

    print(
        "LINE："
    )

    for item in line_frame_map:

        print(
            LINE_DIR /
            item["output_file"]
        )

    print()

    print(
        "帧映射："
    )

    print(
        frame_map_path
    )

    print()

    print(
        "下一步："
    )

    print(
        "python inference_line_frames.py "
        "--path "
        f'"{OUTPUT_DIR}" '
        "--keep_line"
    )

    print()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()