#! python3

import argparse
parser = argparse.ArgumentParser(description='calc similarity image')
parser.add_argument('input1',
                    help='path to image data 1.')
parser.add_argument('input2',
                    help='path to image data 2.', nargs='+')
parser.add_argument('--csv', action='store_true',
                    help='print csv format')
args = parser.parse_args()

import numpy as np
from pathlib import Path

if args.csv:
    print("similarity,source,target")

origin = Path(args.input1)
for origin_path in origin.resolve().parent.glob(origin.name):
    input_path = origin_path
    if not origin.is_absolute():
        input_path = origin_path.relative_to(origin.cwd())
    a = np.load(input_path)

    for filename in map(Path, args.input2):
        for origin_target_path in filename.resolve().parent.glob(filename.name):
            target_path = origin_target_path
            if not filename.is_absolute():
                target_path = origin_target_path.relative_to(filename.cwd())

            b = np.load(target_path)

            # 類似度を計算
            x = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            # 出力
            print(f'{x},"{input_path}","{target_path}"')

