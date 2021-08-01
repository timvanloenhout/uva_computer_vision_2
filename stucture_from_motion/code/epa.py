import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import linear_model
import sys
import operator
import heapq
from scipy import linalg
from pylab import *
import math
from math import pow
import argparse
from pathlib import Path
from scipy.spatial import distance
from numpy.linalg import matrix_rank


def construct_A(p, p_a,):
    p = p.T
    p_a = p_a.T
    A = np.array([p[:,0]*p_a[:,0], p[:,0]*p_a[:,1], p[:,0], p[:, 1]*p_a[:,0],
                  p[:,1]*p_a[:,1], p[:,1], p_a[:,0], p_a[:,1], np.ones((len(p)))]).T

    return(A)

def construct_F(p, p_a):
    if ARGS.normalize == True:
        p, T = normalize_p(p)
        p_a, T_a = normalize_p(p_a)

    A = construct_A(p, p_a)

    # Decompose A
    U, D, Vt = np.linalg.svd(A)
    V = Vt.T

    # Construct F as the column of V corresponding to smallest singular value
    ind = heapq.nsmallest(1, range(len(D)), D.take)
    F = V[:,ind]
    F = F.reshape(3,3)

    # Decompose, set smallest singular value to zero and reconstruct F
    U_f, D_f, V_ft = np.linalg.svd(F)
    ind = heapq.nsmallest(1, range(len(D_f)), D_f.take)
    D_f
    D_f[ind] = 0
    D_f = np.diag(D_f)
    F = U_f.dot(D_f.dot(V_ft))

    # Test epipolar contraint
    # print(np.mean(A.dot(F.reshape(9,1))))

    # Denormalize F
    if ARGS.normalize == True:
        F = (T_a.T).dot(F.dot(T))

    return(F)


def construct_F_ransac(p, p_a):
    max_inliers = []
    for j in range(ARGS.ransac_iters):
        # Get subset of 8 pairs
        ind = np.random.randint(0, p_a.shape[1], 8)
        p_8 = p[:,ind]
        p_a_8 = p_a[:,ind]
        F = construct_F(p_8, p_a_8) # Compute F

        # Count inliers
        d = compute_sampson_distance(p, p_a, F)
        #print(d)
        inliers = [ind for ind, di in enumerate(d) if di < ARGS.sampson_threshold]
        #print(len(inliers))
        if len(inliers) > len(max_inliers):
            max_inliers = inliers

    #print(len(max_inliers))
    # Compute F based using the inliers
    p_inliers = p[:,max_inliers]
    p_a_inliers = p_a[:,max_inliers]
    F = construct_F(p_inliers, p_a_inliers)

    return(F, len(max_inliers))


def normalize_p(p):
    m = np.mean(p, axis=1)
    m = m.reshape(3,1)
    m_x = m[0]
    m_y = m[1]

    d = np.square(p-m)
    d = np.sqrt(np.sum(d[0:2], axis=0))
    d = np.mean(d)

    sqrt2 = np.sqrt(2)

    # Normalization matrix
    T_matrix = np.array([
                [(sqrt2/d), 0, (-m_x*sqrt2/d)],
                [0, (sqrt2/d), (-m_y*sqrt2/d)],
                [0,0,1]])

    # Normalize p
    p_h = T_matrix.dot(p)
    p_h = np.float32(p_h)

    return(p_h, T_matrix)


def compute_sampson_distance(p, p_a, F):
  d = []
  for i in range(p.shape[1]):
    pi = p[:,i]
    pi_a = p_a[:,i]
    num = np.square((pi_a.T).dot(F.dot(pi)))
    den = np.square(F.dot(pi))[0] + np.square(F.dot(pi))[1] + np.square((F.T).dot(pi_a))[0] + np.square((F.T).dot(pi_a))[1]
    di = num + den
    d.append(di)

  return(d)


def get_matches(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # find matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    p = []
    p_a = []
    for i,(m,n) in enumerate(matches):
        if m.distance < ARGS.dist_filter*n.distance: # ratio test as per Lowe's paper
            p_a.append(kp2[m.trainIdx].pt)
            p.append(kp1[m.queryIdx].pt)

    p = np.float32(p).T
    p_a = np.float32(p_a).T

    # convert points to homogeneous coordinates
    w = 1
    p = np.insert(p/w, 2, w, axis=0)
    p_a = np.insert(p_a/w, 2, w, axis=0)

    return(p, p_a)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
    '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def plot_lines(img1, img2, p, p_a, F, n_points_ransac):
    # crawing img1 with epilines corresponding to points in img2
    lines1 = F.dot(p_a).T
    lines1, _ = drawlines(img1,img2,lines1,p[0:2,:].T,p_a[0:2,:].T)

    # crawing img2 with epilines corresponding to points in img1
    lines2 = F.dot(p).T
    lines2,_ = drawlines(img2,img1,lines2,p_a[0:2,:].T,p[0:2,:].T)

    plt.subplot(121),plt.imshow(lines1)
    plt.subplot(122),plt.imshow(lines2)

    Path("../results/fundamental_matrix").mkdir(parents=True, exist_ok=True)
    if ARGS.f_method == 'normal':
        plt.savefig('../results/fundamental_matrix/img{}_{}_{}_n{}.png'
                        .format(ARGS.image_1, ARGS.image_2, ARGS.f_method,
                            ARGS.normalize))

    elif ARGS.f_method == 'ransac':
        plt.savefig('../results/fundamental_matrix/img{}_{}_{}_i{}_t{}_m{}-{}.png'
                        .format(ARGS.image_1, ARGS.image_2, ARGS.f_method,
                            ARGS.ransac_iters, ARGS.sampson_threshold,
                                p.shape[1], n_points_ransac))

    elif ARGS.f_method == 'opencv':
        plt.savefig('../results/fundamental_matrix/img{}_{}_{}.png'
                        .format(ARGS.image_1, ARGS.image_2, ARGS.f_method))


def epa():
    '''
    Eight point algorithm
    '''
    # read images and make them greyscale
    img1 = cv.imread(f'../data/house/frame000000{ARGS.image_n1:02}.png')
    img2 = cv.imread(f'../data/house/frame000000{ARGS.image_n2:02}.png')
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    p, p_a = get_matches(img1, img2)

    n_points_ransac = 0 #variable for printing
    if ARGS.f_method == 'normal':
        F = construct_F(p, p_a)
    elif ARGS.f_method == 'ransac':
        F, n_points_ransac = construct_F_ransac(p, p_a)

    elif ARGS.f_method == 'opencv':
        F,_ = cv.findFundamentalMat(p[0:2,:].T,p_a[0:2,:].T, cv.FM_LMEDS)

    plot_lines(img1, img2, p, p_a, F, n_points_ransac)



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--img_n1', default=1, type=int,
                        help='fist image')
    PARSER.add_argument('--img_n2', default=2, type=int,
                        help='second image')
    PARSER.add_argument('--f_method', default='ransac', type=str,
                        help='method for constructing fundamental matrix',
                        choices=['normal', 'ransac', 'opencv'])
    PARSER.add_argument('--normalize', default=True, type=bool,
                        help='sse normalized points')
    PARSER.add_argument('--ransac_iters', default=200, type=int,
                        help='number of ransac iterations')
    PARSER.add_argument('--sampson_threshold', default=0.0001, type=int,
                        help='sampson_threshold for ransac distances')
    PARSER.add_argument('--dist_filter', default=0.25, type=float,
                        help='initial points filtering')


    ARGS = PARSER.parse_args()
    epa()
