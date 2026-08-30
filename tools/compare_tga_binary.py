from pathlib import Path
import struct
import hashlib
import argparse


def read_tga_header(data):
    if len(data) < 18:
        return None

    return {
        "id_length": data[0],
        "color_map_type": data[1],
        "image_type": data[2],
        "color_map_first": struct.unpack_from("<H", data, 3)[0],
        "color_map_length": struct.unpack_from("<H", data, 5)[0],
        "color_map_depth": data[7],
        "x_origin": struct.unpack_from("<H", data, 8)[0],
        "y_origin": struct.unpack_from("<H", data, 10)[0],
        "width": struct.unpack_from("<H", data, 12)[0],
        "height": struct.unpack_from("<H", data, 14)[0],
        "pixel_depth": data[16],
        "image_descriptor": data[17],
    }


def print_header(name, h):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if h is None:
        print("无法读取 TGA Header")
        return

    for k, v in h.items():
        print(f"{k:22s}: {v}")


def hex_dump(data, start, length=128):
    end = min(start + length, len(data))

    for offset in range(start, end, 16):
        chunk = data[offset:min(offset + 16, end)]

        hex_part = " ".join(
            f"{x:02X}" for x in chunk
        )

        ascii_part = "".join(
            chr(x) if 32 <= x < 127 else "."
            for x in chunk
        )

        print(
            f"{offset:08X}  "
            f"{hex_part:<47}  "
            f"{ascii_part}"
        )


def find_tga_footer(data):
    """
    标准 TGA 2.0 footer：

    Offset 0:
        Extension Area Offset
        4 bytes

    Offset 4:
        Developer Directory Offset
        4 bytes

    Offset 8:
        Signature
        'TRUEVISION-XFILE'
    """

    if len(data) < 26:
        return None

    footer = data[-26:]

    signature = footer[8:24]

    if signature == b"TRUEVISION-XFILE":
        ext_offset = struct.unpack_from("<I", footer, 0)[0]
        dev_offset = struct.unpack_from("<I", footer, 4)[0]

        return {
            "extension_offset": ext_offset,
            "developer_offset": dev_offset,
            "signature": signature.decode(
                "ascii",
                errors="replace"
            )
        }

    return None


def analyze(path):

    path = Path(path)

    data = path.read_bytes()

    header = read_tga_header(data)

    footer = find_tga_footer(data)

    sha256 = hashlib.sha256(data).hexdigest()

    return {
        "path": str(path),
        "size": len(data),
        "sha256": sha256,
        "header": header,
        "footer": footer,
        "data": data,
    }


def compare_prefix(a, b):

    max_len = min(
        len(a),
        len(b)
    )

    same = 0

    for i in range(max_len):

        if a[i] == b[i]:
            same += 1
        else:
            break

    return same


def main():

    parser = argparse.ArgumentParser(
        description="Binary compare two TGA files."
    )

    parser.add_argument(
        "--a",
        required=True,
        help="原始 PaintMan TGA"
    )

    parser.add_argument(
        "--b",
        required=True,
        help="Python 生成的 TGA"
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    A = analyze(args.a)
    B = analyze(args.b)

    print()
    print("=" * 70)
    print("TGA 二进制结构比较")
    print("=" * 70)

    print()
    print("A:")
    print(A["path"])

    print("B:")
    print(B["path"])

    # --------------------------------------------------
    # 基本信息
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("文件基本信息")
    print("=" * 70)

    print(
        "A size   :",
        A["size"],
        "bytes"
    )

    print(
        "B size   :",
        B["size"],
        "bytes"
    )

    print(
        "A SHA256 :",
        A["sha256"]
    )

    print(
        "B SHA256 :",
        B["sha256"]
    )

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    print_header(
        "A - 原始 PaintMan TGA Header",
        A["header"]
    )

    print_header(
        "B - Python TGA Header",
        B["header"]
    )

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("TGA Footer")
    print("=" * 70)

    print()
    print("A footer:")
    print(A["footer"])

    print()
    print("B footer:")
    print(B["footer"])

    # --------------------------------------------------
    # 文件前缀
    # --------------------------------------------------

    prefix = compare_prefix(
        A["data"],
        B["data"]
    )

    print()
    print("=" * 70)
    print("二进制共同前缀")
    print("=" * 70)

    print(
        "完全相同的连续字节:",
        prefix
    )

    print(
        "A 文件大小:",
        len(A["data"])
    )

    print(
        "B 文件大小:",
        len(B["data"])
    )

    # --------------------------------------------------
    # Header hex
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("A Header HEX")
    print("=" * 70)

    hex_dump(
        A["data"],
        0,
        256
    )

    print()
    print("=" * 70)
    print("B Header HEX")
    print("=" * 70)

    hex_dump(
        B["data"],
        0,
        256
    )

    # --------------------------------------------------
    # 文件尾部
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("A 文件尾部")
    print("=" * 70)

    start_a = max(
        0,
        len(A["data"]) - 512
    )

    hex_dump(
        A["data"],
        start_a,
        512
    )

    print()
    print("=" * 70)
    print("B 文件尾部")
    print("=" * 70)

    start_b = max(
        0,
        len(B["data"]) - 512
    )

    hex_dump(
        B["data"],
        start_b,
        512
    )

    # --------------------------------------------------
    # 输出报告
    # --------------------------------------------------

    if args.output:

        out = Path(args.output)

        out.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            out,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "TGA 二进制结构比较\n"
            )

            f.write("=" * 70 + "\n\n")

            f.write(
                f"A: {A['path']}\n"
            )

            f.write(
                f"B: {B['path']}\n\n"
            )

            f.write(
                f"A size: {A['size']}\n"
            )

            f.write(
                f"B size: {B['size']}\n\n"
            )

            f.write(
                f"A SHA256: {A['sha256']}\n"
            )

            f.write(
                f"B SHA256: {B['sha256']}\n\n"
            )

            f.write(
                f"A header: {A['header']}\n"
            )

            f.write(
                f"B header: {B['header']}\n\n"
            )

            f.write(
                f"A footer: {A['footer']}\n"
            )

            f.write(
                f"B footer: {B['footer']}\n\n"
            )

            f.write(
                f"共同前缀: {prefix}\n"
            )

        print()
        print(
            "[OK] 报告保存：",
            out
        )


if __name__ == "__main__":
    main()