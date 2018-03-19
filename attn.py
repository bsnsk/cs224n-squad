import numpy as np
import matplotlib.pyplot as plt

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
    attn = np.squeeze(attn)
    # print "attn.shape={}".format(attn.shape)
    if len(attn.shape) > 1:
        norms = np.linalg.norm(attn, axis=1)  # size (context_len)
    else:
        norms = attn
    L = len(sample[0].split(' ')) - 1
    # print "L={}".format(L)
    plt.clf()
    xs = range(1, L+1)
    ys = norms[:L]
    plt.plot(xs, ys, '.-')
    plt.title(sample[1] + sample[2])
    plt.xlabel("position in context")
    plt.ylabel("attention norm")
    plt.show()


plotAttn(self_attn[0], samples[0])
