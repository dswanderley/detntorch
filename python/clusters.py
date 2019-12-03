
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.datasets import OvaryDataset
from torch.utils.data import DataLoader


N_CLUSTERS = 6

path_im = "../datasets/ovarian/im/test"
path_gt = "../datasets/ovarian/gt/test"

widths = []
heights = []

# pre-set
dataset = OvaryDataset(im_dir=path_im,
                        gt_dir=path_gt,
                        clahe=False, transform=False,
                        ovary_inst=True,
                        out_tuple=True)
# Loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=dataset.collate_fn_yolo)

# iterate
for _, (fname, img, targets) in enumerate(data_loader):
    # Load data

    img_size = img.shape[-1]

    # read each image in batch
    for idx, lbl, cx, cy, w, h in targets:
        widths.append(w.item() * img_size)
        heights.append(h.item() * img_size)

# Compute clusters
data = np.array( [widths, heights] )
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data.transpose())

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for xy, cl in zip(data.transpose(), kmeans.labels_):
    x = xy[0]
    y = xy[1]
    c = colors[cl]
    plt.scatter(x, y, c=c, marker=".")

for i in range(len(kmeans.cluster_centers_)):
    x = round(kmeans.cluster_centers_[i][0])
    y = round(kmeans.cluster_centers_[i][1])
    area = x*y

    print('width: ' + str(x) , 'height: ' + str(y), 'area: ' + str(area))

    plt.scatter(x, y, c='k', marker="*")

plt.axis([0, 400, 0, 300])

plt.show()

print('')