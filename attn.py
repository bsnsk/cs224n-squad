import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

bidaf_attn_c2q = np.load("./log/bidaf-attn-c2q.log")
bidaf_attn_q2c = np.load("./log/bidaf-attn-q2c.log")
co_attn_c2q = np.load("./log/co-attn-c2q.log")
co_attn_q2c = np.load("./log/co-attn-q2c.log")
self_attn = np.load("./log/self-attn.log")

with open("./log/samples.log", "r") as f:
    lines = f.readlines()
    samples = [(lines[i*4], lines[i*4+1], lines[i*4+2]) for i in range(10)]


# attn: size [context_len, *]
def plotAttn(attn, sample):
    context = "norman architecture typically stands out as a new stage in the architectural history of the regions they subdued . they spread a unique romanesque idiom to england and italy , and the encastellation of these regions with keeps in their north french style fundamentally altered the military landscape . their style was characterised by rounded arches , particularly over windows and doorways , and massive proportions ."
    attn = np.squeeze(attn)
    print "attn.shape={}".format(attn.shape)
    L = len(sample[0].split(' ')) - 1
    # print "L={}".format(L)
    data = attn[:L, :L]
    # fig, axes = plt.subplot(1, 1)
    # xs = range(1, L+1)
    # ys = norms[:L]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = colors.ListedColormap([
        (1. - x * 0.01, 1. - x * 0.005, 1. - x * 0.01)
        for x in range(100)
    ])
    ax.set_xticklabels(context.split(" "), rotation=90)
    ax.set_yticklabels(context.split(" "))
    ax.tick_params(axis='both', width=1, top='off', labelsize=6)
    plt.xticks(range(0, L))
    plt.yticks(range(0, L))
    # plt.xlabel(context.split(" "))
    # plt.ylabel(context.split(" "))
    # plt.xticks(rotation=90)
    # plt.pcolor(attn[:L, :L])
    ax.imshow(attn[:L, :L], cmap=cmap, interpolation='nearest')
    # plt.plot(xs, ys, '.-')
    # plt.title(sample[1] + sample[2])
    # plt.xlabel("position in context")
    # plt.ylabel("attention norm")
    plt.show()


plotAttn(self_attn[0], samples[0])
