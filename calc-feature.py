#! python3

import argparse
parser = argparse.ArgumentParser(description='calc image feature')
parser.add_argument('output',
                    help='path to output vector dir.')
parser.add_argument('path',
                    help='path to image data.', nargs='+')
args = parser.parse_args()

import numpy as np
import os
from pathlib import Path
from progressbar import progressbar
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.preprocessing.image as Image
from keras.models import Model

ext = ['.jpg', '.jpeg', '.bmp', '.png', '.ppm']

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=None
)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

for filename in progressbar(map(Path, args.path)):
    for image_path in filename.resolve().parent.glob(filename.name):
        if image_path.suffix.lower() not in ext:
            continue

        # サイズを(224, 224)に変換し画像読み込み
        image = Image.load_img(image_path, target_size=(224, 224))

        # 配列に変換
        x = Image.img_to_array(image)
        # 軸を指定して次元を増やす
        x = np.expand_dims(x, axis=0)
        # カラーのチャネルの順番をRGB→BGR
        x = preprocess_input(x)
        y = intermediate_layer_model.predict(x)

        # ファイルに出力
        output_path = Path(args.output).joinpath(Path(filename).stem)
        np.save(output_path, y.flatten())

