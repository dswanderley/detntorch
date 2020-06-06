
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.datasets import OvaryDataset
from torch.utils.data import DataLoader


N_CLUSTERS = 6
ovary = True

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

im_list = ["../datasets/ovarian/im/test", "../datasets/ovarian/im/train", "../datasets/ovarian/im/val"]
gt_list = ["../datasets/ovarian/gt/test", "../datasets/ovarian/gt/train", "../datasets/ovarian/gt/val"]

widths = []
heights = []

# Read all datasets
for (path_im, path_gt) in zip(im_list, gt_list):

    # pre-set
    dataset = OvaryDataset(im_dir=path_im,
                            gt_dir=path_gt,
                            clahe=False, transform=False,
                            ovary_inst=ovary)
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

for xy, cl in zip(data.transpose(), kmeans.labels_):
    x = xy[0]
    y = xy[1]
    c = colors[cl]
    plt.scatter(x, y, c=c, marker=".")

centers = [ [c[0], c[1], c[0]*c[1]] for c in kmeans.cluster_centers_]
centers = np.array(centers)
centers = np.sort(centers.view('i8,i8,i8'), order=['f1'], axis=0).view(np.float)

# outputs
print('N_CLUSTERS: ', N_CLUSTERS)
print('ovary: ', ovary)

# Print clusters
for w, h, area in centers:
    print('width: ' + str( round(w) ) , 'height: ' + str( round(h) ), 'area: ' + str( round(area) ))
    plt.scatter(w, h, c='k', marker="*")

plt.axis([ 0, max(widths)+5, 0, max(heights)+5 ])
plt.show()