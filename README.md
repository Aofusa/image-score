画像のスコア化
---

適当な分析を行い画像を数値化  
適当な指標でいい感じに分類する  

## 使い方いろいろ
 `calc-feature.py` で好みの画像をいくつかベクトルに変換した後、
 `statistic.py` で平均のベクトルを作成。  
その後、作った好みの画像の平均のベクトルを使って `img-similarity.py` で
いろいろな画像と比較すると、その画像が自分の好みかどうかがだいたいわかるかもしれない  

### calc-feature.py  
画像をベクトルに変換して保存  
```sh
python3 calc-feature.py <対象の画像> <保存先のディレクトリ>
```

- 例  
```sh
python3 calc-feature.py images/* vectors
```


### img-similarity.py  
保存された画像のベクトルと画像との類似度を計算して画面に出力  
```sh
python3 imp-similarity.py <画像のベクトル> <類似度を計算したい画像>
```

- 例  
```sh
python3 imp-similarity.py vectors/*.npy images/*
```


### similarity.py  
 `img-similarity.py` のベクトル同士版  
```sh
python3 imp-similarity.py <画像のベクトル> <類似度を計算したい画像のベクトル>
```

- 例  
```sh
python3 imp-similarity.py vectors/*.npy vectors/*
```


### plot.py  
 保存された画像のベクトルの相関を分類しプロットする  
```sh
python3 plot.py <画像のベクトル> [-n <分類する数>] [-pca|-lda|-kernel-pca]
```

- 例  
```sh
python3 plot.py vectors/*
```


### statistic.py  
作成したベクトルを使って統計ちっくなことに使えそうなあれやこれやをいれた  
そのままでは使えない  
インタープリターにコピペして使うよう、みたいな感じ  
