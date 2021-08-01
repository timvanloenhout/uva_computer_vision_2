import numpy as np

def read_pcd(fname, noise_threshold, fname_normal=None):
    """ read in pcd files """
    data = []
    version = 0
    width = 0
    height = 0
    points = 0
    final = []

    with open(fname, 'r') as f, open(fname_normal, 'r') as fn:
        lines = f.readlines()
        norm_lines = fn.readlines()
        for i, l in enumerate(lines):
            l = l.split(' ')
            ln = norm_lines[i].split(' ')
            l[-1] = l[-1].strip('\n')
            ln[-1] = ln[-1].strip('\n')
            if not l[0].isalpha() and not l[0] == '#':
                l = [float(i) for i in l]
                ln = [float(j) for j in ln]
                data.append(l[:-1])
                data[-1].extend(ln[:-1])

    pcd = np.array(data)
    pcd = pcd[pcd[:, 2] < noise_threshold]
    norm = pcd[:, 3:]
    pcd = pcd[:, :3]
    return pcd, norm

