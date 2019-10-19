#! python3

import argparse
parser = argparse.ArgumentParser(description='calc similarity image')
parser.add_argument('input1',
                    help='path to numpy data.')
parser.add_argument('input2',
                    help='path to image.', nargs='+')
parser.add_argument('--csv', action='store_true',
                    help='print csv format')
args = parser.parse_args()

import numpy as np
from pathlib import Path
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.preprocessing.image as Image
from keras.models import Model

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=None
)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

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

            # サイズを(224, 224)に変換し画像読み込み
            image = Image.load_img(target_path, target_size=(224, 224))

            # 配列に変換
            x = Image.img_to_array(image)
            # 軸を指定して次元を増やす
            x = np.expand_dims(x, axis=0)
            # 各画素値から平均値 (103.939, 116.779, 123.68) を引く
            # カラーのチャネルの順番をRGB→BGR
            x = preprocess_input(x)
            y = intermediate_layer_model.predict(x)

            # 次元を画像1枚分に削減
            b = y.flatten()

            # 類似度を計算
            x = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            # 出力
            print(f'{x},"{input_path}","{target_path}"')

