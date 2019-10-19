#! python3

import argparse
parser = argparse.ArgumentParser(description='calc similarity image')
parser.add_argument('path',
                    help='path to data.', nargs='+')
args = parser.parse_args()

import numpy as np
from pathlib import Path
from progressbar import progressbar

ext = ['.npy', '.npz']

# numpy 配列の読み込み
data = []
for filename in progressbar(map(Path, args.path)):
    for data_path in filename.resolve().parent.glob(filename.name):
        if data_path.suffix.lower() not in ext:
            continue
        x = np.load(data_path)
        data.append(x)
data = np.array(data)

# 平均
data_mean = data.mean(axis=0)
# 最大
data_max = data.max(axis=0)
# 最小
data_min = data.min(axis=0)
# 距離
data_distance = np.linalg.norm(data, axis=0)
# 標準偏差
data_std = data.std(axis=0)
# 分散
data_var = data.var(axis=0)
# 標準化
data_zscore = (data - data.mean(axis=0)) / data.std(axis=0)
# 正規化
data_normalize = data / np.linalg.norm(data, axis=0)
