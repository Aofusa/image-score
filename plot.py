#! python3

import numpy as np
import pandas as pd
from pandas import plotting
import sklearn
import matplotlib.pyplot as plt

def get_arg():
    import argparse
    parser = argparse.ArgumentParser(description='plot image vector')
    parser.add_argument('path',
                        help='path to image vector data.', nargs='+')
    parser.add_argument('-n', '--number-clusters', type=int,
                        help='number of clustering. (default: 3)', default=3)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pca', action='store_true',
                       help='Plot using PCA (default)')
    group.add_argument('--lda', action='store_true',
                       help='Plot using LDA')
    group.add_argument('--kernel-pca', action='store_true',
                       help='Plot using KernelPCA')
    args = parser.parse_args()
    return args


def initialize():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']


def smooth_color(t, t_max):
    v = (float) (t / t_max)
    if 0 <= v and v <= 1.0/6.0:
    # t1 #FF0000 -> #FFFF00
        r = 0xff
        g = (int) (0xff * (v-0) / (1.0/6.0))
        b = 0x00
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)
    elif 1.0/6.0 < v and v <= 2.0/6.0:
    # t2 #FFFF00 -> #00FF00
        r = (int) (0xff * (1.0 - (v-(1.0/6.0)) / (1.0/6.0)))
        g = 0xff
        b = 0x00
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)
    elif 2.0/6.0 < v and v <= 3.0/6.0:
    # t3 #00FF00 -> #00FFFF
        r = 0x00
        g = 0xff
        b = (int) (0xff * (v-(2.0/6.0)) / (1.0/6.0))
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)
    elif 3.0/6.0 < v and v <= 4.0/6.0:
    # t4 #00FFFF -> #0000FF
        r = 0x00
        g = (int) (0xff * (1.0 - (v-(3.0/6.0)) / (1.0/6.0)))
        b = 0xff
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)
    elif 4.0/6.0 < v and v <= 5.0/6.0:
    # t5 #0000FF -> #FF00FF
        r = (int) (0xff * (v-(4.0/6.0)) / (1.0/6.0))
        g = 0x00
        b = 0xff
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)
    elif 5.0/6.0 < v and v <= 6.0/6.0:
    # t6 #FF00FF -> #FF0000
        r = 0xff
        g = 0x00
        b = (int) (0xff * (1.0 - (v-(5.0/6.0)) / (1.0/6.0)))
        return '#{r:02x}{g:02x}{b:02x}'.format(r=r,g=g,b=b)


def generate_color(number):
    color = []
    for i in range(number):
        color.append(smooth_color(i, number))
    return color


def load_ndarray(path):
    from pathlib import Path
    from progressbar import progressbar

    # numpy 配列の読み込み
    data = []
    data_name = []
    prog = progressbar(map(Path, path))
    for filename in prog:
        for data_path in filename.resolve().parent.glob(filename.name):
            data_name.append(data_path.name)
            x = np.load(data_path)
            data.append(x)

    data = np.array(data)
    prog.close()

    return (data, data_name)


def make_dataframe(data):
    # pandas DataFrame に変換
    df = pd.DataFrame(data)

    # 行列の標準化
    dfs = df.iloc[:, :].apply(lambda x: x/np.linalg.norm(x), axis=0).fillna(0)
    # dfs.head()

    return dfs


def plot_lda(dfs, data_name, n_clusters):
    # K-means クラスタリング
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(dfs)

    # # 色分けした Scatter Matrix を描く。
    color_codes = generate_color(n_clusters)
    colors = [color_codes[x] for x in kmeans_model.labels_]

    # 線形判別分析の実行
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    feature = lda.fit_transform(dfs, kmeans_model.labels_)

    # 主成分得点
    # pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(2)]).head()

    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for x, y, name in zip(feature[:, 0], feature[:, 1], data_name):
        plt.text(x, y, name, alpha=0.5, size=15)

    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=colors)
    plt.grid()
    plt.show()



def plot_kernel_pca(dfs, data_name, n_clusters):
    # カーネル主成分分析の実行
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=20.0)
    feature = kpca.fit_transform(dfs)

    # 主成分得点
    # pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(2)]).head()

    # K-means クラスタリング
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(feature)

    # # 色分けした Scatter Matrix を描く。
    color_codes = generate_color(n_clusters)
    colors = [color_codes[x] for x in kmeans_model.labels_]

    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for x, y, name in zip(feature[:, 0], feature[:, 1], data_name):
        plt.text(x, y, name, alpha=0.5, size=15)

    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=colors)
    plt.grid()
    plt.show()



def plot_pca(dfs, data_name, n_clusters):
    #主成分分析の実行
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    feature = pca.fit(dfs)

    # データを主成分空間に写像
    feature = pca.transform(dfs)

    # 主成分得点
    # pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(2)]).head()

    # K-means クラスタリング
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(feature)

    # # 色分けした Scatter Matrix を描く。
    color_codes = generate_color(n_clusters)
    colors = [color_codes[x] for x in kmeans_model.labels_]

    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for x, y, name in zip(feature[:, 0], feature[:, 1], data_name):
        plt.text(x, y, name, alpha=0.5, size=15)

    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=colors)
    plt.grid()
    plt.show()


if __name__=='__main__':
    args = get_arg()
    initialize()

    data, data_name = load_ndarray(args.path)
    dfs = make_dataframe(data)

    if args.pca:
        plot_pca(dfs, data_name, args.number_clusters)
    elif args.lda:
        plot_lda(dfs, data_name, args.number_clusters)
    elif args.kernel_pca:
        plot_kernel_pca(dfs, data_name, args.number_clusters)
    else:
        plot_pca(dfs, data_name, args.number_clusters)

