import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def tsne_visualization(data_dir='/apdcephfs/private_xiaohanxing/STAC_CRD_MT_codes/feature_visualization/'):
    data_path = data_dir + 'train_features/embed4_labels.csv'
    features_labels = np.array(pd.read_csv(data_path, header=None))[:, :].astype(float)
    print(features_labels.shape)

    feature = features_labels[:, :-1]
    label = features_labels[:, -1].astype(int)
    # print(feature.shape)
    # print(label)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feature)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(data_dir + 'tsne_maps/train_embed4.jpg')
    plt.show()


tsne_visualization()
