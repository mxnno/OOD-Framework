import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def draw(x, y):
    from matplotlib.colors import ListedColormap
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    print("TSNE: fitting start...")
    tsne = TSNE(2, n_jobs=4, perplexity=30)
    Y = tsne.fit_transform(x)

    # matplotlib_axes_logger.setLevel('ERROR')
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','open']
    id_to_label = {i: label for i, label in enumerate(labels) }
    y_true = pd.Series(y)
    plt.style.use('ggplot')
    n_class = y_true.unique().shape[0]
    colors = ( 'gray','lightgreen', 'plum','DarkMagenta','SkyBlue','PaleTurquoise','DeepPink','Gold','Orange','Brown','DarkKhaki')

    fig, ax = plt.subplots(figsize=(9, 6), )
    la = [i for i in range(n_class)]
    la = sorted(la,reverse=True)
    cmap = ListedColormap(colors)
    for idx, label in enumerate(la):
        ix = y_true[y_true==label].index
        x = Y[:, 0][ix]
        y = Y[:, 1][ix]
        ax.scatter(x, y, c=cmap(idx), label=id_to_label[label], alpha=0.5)

    # Shrink current axis by 20%
    ax.set_title('proto_loss')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_curve(points):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    plt.grid(alpha=0.4)
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', 'H', 'v', '>', 'h', 'H', 'v', '>', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = np.arange(-0.02, len(centers[0]) + 0.01).astype(dtype=np.str)
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
    
    plt.xlim(0, 20, 1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Decision Boundary $\Delta$', fontsize=12)
    plt.legend()
    plt.title('50% Known Classes on StackOverflow')
    plt.show()
    plt.savefig('curve.pdf')

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels):
        
        logits = euclidean_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1) 
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()
        
        return loss, delta 