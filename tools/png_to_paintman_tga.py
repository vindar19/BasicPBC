from pathlib import Path
from PIL import Image
import argparse


def png_to_tga(src, dst):
    src = Path(src)
    dst = Path(dst)

    im = Image.open(src)

    # --------------------------------------------------
    # 强制 RGB
    #
    # 不使用 Alpha
    # 不做透明化
    # 不修改任何 RGB 数值
    # --------------------------------------------------

    rgb = im.convert("RGB")

    # --------------------------------------------------
    # 保存为标准 24-bit TGA
    #
    # Pillow:
    #   RGB -> 24bit TGA
    #   compression=tga_rle
    # --------------------------------------------------

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rgb.save(
        dst,
        format="TGA",
        compression="tga_rle"
    )

    print(
        f"[OK] {src.name} -> {dst.name}"
    )

    print(
        f"     size = {rgb.size}"
    )

    print(
        f"     mode = {rgb.mode}"
    )


def convert_directory(src_dir, dst_dir):

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pngs = sorted(
        src_dir.glob("*.png")
    )

    if not pngs:
        print(
            "[ERROR] 没有找到 PNG：",
            src_dir
        )
        return

    print("=" * 70)
    print("BasicPBC PNG -> PaintMan TGA")
    print("=" * 70)

    print()
    print("输入:")
    print(src_dir)

    print()
    print("输出:")
    print(dst_dir)

    print()
    print("PNG 数量:", len(pngs))

    print()
    print("转换规则:")
    print("  RGB 原值完全保留")
    print("  255,255,255 保留为纯白")
    print("  近白色不透明化、不修改")
    print("  不使用 Alpha")
    print("  输出 24-bit TGA")
    print("  使用 TGA RLE 压缩")

    print()

    for png in pngs:

        tga = dst_dir / (
            png.stem + ".tga"
        )

        png_to_tga(
            png,
            tga
        )

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description="Convert BasicPBC PNG results to PaintMan TGA."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="PNG 文件或 PNG 文件夹"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出 TGA 文件或文件夹"
    )

    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    # 单文件
    if src.is_file():

        if dst.suffix.lower() != ".tga":
            dst = dst / (
                src.stem + ".tga"
            )

        png_to_tga(
            src,
            dst
        )

    # 文件夹
    elif src.is_dir():

        convert_directory(
            src,
            dst
        )

    else:

        print(
            "[ERROR] 输入不存在：",
            src
        )


if __name__ == "__main__":
    main()