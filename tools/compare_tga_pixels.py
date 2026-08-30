from PIL import Image
import numpy as np
import argparse


def load(path):
    im = Image.open(path)
    rgb = np.array(im.convert("RGB"))

    print()
    print("FILE:", path)
    print("format :", im.format)
    print("mode   :", im.mode)
    print("size   :", im.size)
    print("shape  :", rgb.shape)

    return rgb


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)

    args = parser.parse_args()

    print("=" * 70)
    print("PaintMan TGA / Python TGA 像素级比较")
    print("=" * 70)

    a = load(args.a)
    b = load(args.b)

    if a.shape != b.shape:
        print()
        print("[ERROR] 图像尺寸不同")
        return

    # --------------------------------------------------
    # 精确比较
    # --------------------------------------------------

    diff = np.abs(
        a.astype(np.int16)
        -
        b.astype(np.int16)
    )

    different = np.any(
        diff != 0,
        axis=2
    )

    print()
    print("=" * 70)
    print("比较结果")
    print("=" * 70)

    print(
        "MAX RGB DIFF     =",
        diff.max()
    )

    print(
        "MEAN RGB DIFF    =",
        diff.mean()
    )

    print(
        "DIFFERENT PIXELS =",
        different.sum()
    )

    total = different.size

    print(
        "TOTAL PIXELS     =",
        total
    )

    print(
        "DIFFERENT %      =",
        different.sum() / total * 100,
        "%"
    )

    # --------------------------------------------------
    # 每个通道
    # --------------------------------------------------

    print()
    print("Channel differences:")

    for i, name in enumerate(
        ["R", "G", "B"]
    ):

        d = diff[:, :, i]

        print(
            f"{name}: "
            f"max={d.max()} "
            f"mean={d.mean():.6f} "
            f"pixels={(d != 0).sum()}"
        )

    # --------------------------------------------------
    # 像素统计
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("关键颜色统计")
    print("=" * 70)

    colors = [
        (255, 255, 255),
        (0, 0, 0),
    ]

    for color in colors:

        ma = np.all(
            a == color,
            axis=2
        )

        mb = np.all(
            b == color,
            axis=2
        )

        print()
        print(color)

        print(
            " A:",
            ma.sum()
        )

        print(
            " B:",
            mb.sum()
        )

    # --------------------------------------------------
    # 差异区域边界
    # --------------------------------------------------

    if different.any():

        ys, xs = np.where(
            different
        )

        print()
        print("=" * 70)
        print("差异区域")
        print("=" * 70)

        print(
            "X:",
            xs.min(),
            "~",
            xs.max()
        )

        print(
            "Y:",
            ys.min(),
            "~",
            ys.max()
        )

    else:

        print()
        print("[OK] 两张图片 RGB 像素完全一致。")


if __name__ == "__main__":
    main()