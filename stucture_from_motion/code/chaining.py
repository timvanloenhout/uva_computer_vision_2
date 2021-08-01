import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import pandas as pd
import argparse
from pathlib import Path

def get_flann_matches(img_n1, img_n2):
    img1 = cv.imread(f'../data/house/frame000000{img_n1:02}.png')
    img2 = cv.imread(f'../data/house/frame000000{img_n2:02}.png')

    grey1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    grey2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # perform sift to get keypoints and descriptors
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    return matches, kp1, kp2

def get_matches(img_n1, img_n2, t):
    img1 = cv.imread(f'../data/house/frame000000{img_n1:02}.png')
    img2 = cv.imread(f'../data/house/frame000000{img_n2:02}.png')
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # perform sift to get keypoints and descriptors
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # match descriptors (using L1 norm now, because it's the only one that worked...)
    matcher = cv.BFMatcher(cv.NORM_L1)
    matches = matcher.match(des1, des2)

    # only keep good matcjes (why? idk)
    matches.sort(key=lambda x: x.distance)
    num_good_matches = int(len(matches) * t)
    matches = matches[:num_good_matches]

    return matches, kp1, kp2

def set_points(pvm, img_n1, img_n2, feature_point, p, p_a):
    pvm.loc[f'x{img_n1}', feature_point] = p[-1][0]
    pvm.loc[f'y{img_n1}', feature_point] = p[-1][1]
    pvm.loc[f'x{img_n2}', feature_point] = p_a[-1][0]
    pvm.loc[f'y{img_n2}', feature_point] = p_a[-1][1]

    return pvm

def get_pvm(pvm, img_n1, img_n2, p, p_a):
    found = False
    for feature_point, point in enumerate(zip(pvm.loc[f'x{img_n1}'], pvm.loc[f'y{img_n1}'])):
        if nearby(p[-1], point, ARGS.nearby_filter):
            found = True
            pvm.ix[f'x{img_n2}', feature_point] = p_a[-1][0]
            pvm.ix[f'y{img_n2}', feature_point] = p_a[-1][1]
            break

    # when no new points are found, add a new column
    if not found:
        feature_point = f'{pvm.shape[1] + 1}'
        pvm = set_points(pvm, img_n1, img_n2, feature_point, p, p_a)

    return pvm

def nearby(points1, points2, t):
    x1, y1 = points1
    x2, y2 = points2
    close = x1 - t <= x2 and x2 <= x1 + t and y1 - t <= y2 and y2 <= y1 + t
    return close

def test_pvm():
    with open('PointViewMatrix.txt', 'r') as f:
        bigboi = np.zeros((202, 215))
        for i, line in enumerate(f):
            print(i)
            line = np.array(line.split(' '))
            print(line.shape)
            bigboi[i, :] = line.squeeze()

        bigboi = np.where(bigboi > 0, 1, 0)

        plt.imshow(bigboi, aspect='auto', cmap='gray')
        plt.title('Point-View Matrix from PointViewMatrix.txt')
        plt.xlabel('Feature Points (M)')
        plt.ylabel('Images (2N)')
        Path("../results/chaining/images").mkdir(parents=True, exist_ok=True)
        plt.savefig('../results/chaining/images/pvm_from_PointViewMatrix_norm.png')
        plt.show()


def show_off():
    Path("../results/chaining/pvm").mkdir(parents=True, exist_ok=True)
    pvm = pd.read_csv(f'../results/chaining/pvm/pvm_{ARGS.match_method}_{ARGS.dist_filter}_{ARGS.nearby_filter}.csv', index_col=0).values

    norm = np.where(np.isnan(pvm), 1, 0)
    plt.imshow(pvm, aspect='auto', cmap='gray')
    plt.title('Point-View Matrix')
    plt.xlabel('Feature Points (M)')
    plt.ylabel('Images (2N)')

    plt.imshow(norm, aspect='auto', cmap='gray')
    plt.title('Point-View Matrix')
    plt.xlabel('Feature Points (M)')
    plt.ylabel('Images (2N)')
    Path("../results/chaining/images").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../results/chaining/images/normalized_pvm_{ARGS.match_method}_{ARGS.dist_filter}_{ARGS.nearby_filter}.pdf')
    plt.show()


def chaining():
    pvm = pd.DataFrame()

    # iterate over all images, compare 1-2, 2-3, 48-49, 49-1
    for img_n1 in range(1,50):
        if img_n1 + 1 < 50:
            img_n2 = img_n1 + 1
        else:
            img_n2 = 1

        print(f'Image {img_n1} and {img_n2}')

        p = []
        p_a = []

        if ARGS.match_method == 'bf':
            matches, kp1, kp2 = get_matches(img_n1, img_n2, ARGS.dist_filter)

            # iterate over matches and create pvm
            for i, match in enumerate(matches):
                p.append(kp1[match.queryIdx].pt)
                p_a.append(kp2[match.trainIdx].pt)

                # very first iteration, create first column
                if pvm.shape[1] == 0:
                    feature_point = f'{i + 1}'
                    pvm = set_points(pvm, img_n1, img_n2, feature_point, p, p_a)
                else:
                    pvm = get_pvm(pvm, img_n1, img_n2, p, p_a)

        elif ARGS.match_method == 'flann':
            matches, kp1, kp2 = get_flann_matches(img_n1, img_n2)

            for i,(m, n) in enumerate(matches):
                # ratio test as per Lowe's paper
                if m.distance < ARGS.dist_filter*n.distance:
                    p_a.append(kp2[m.trainIdx].pt)
                    p.append(kp1[m.queryIdx].pt)

                    # very first iteration, create first column
                    if pvm.shape[1] == 0:
                        feature_point = f'{i + 1}'
                        pvm = set_points(pvm, img_n1, img_n2, feature_point, p, p_a)
                    else:
                        pvm = get_pvm(pvm, img_n1, img_n2, p, p_a)

    Path("../results/chaining/pvm").mkdir(parents=True, exist_ok=True)
    pvm.to_csv(f'../results/chaining/pvm/pvm_{ARGS.match_method}_{ARGS.dist_filter}_{ARGS.nearby_filter}.csv')
    print(pvm.shape)
    pvm = pvm.values
    print(pvm.shape)

    if ARGS.vizualize:
        show_off()

    return pvm

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--vizualize', default=True, type=bool,
                        help='whether to visualize the result')
    PARSER.add_argument('--match_method', default='bf', type=str,
                        help='which method to use for matching feature points', choices=['bf', 'flann'])
    PARSER.add_argument('--dist_filter', default=0.25, type=float,
                        help='initial points filtering')
    PARSER.add_argument('--nearby_filter', default=10, type=int,
                        help='threshold for determining whether two points are similar')

    ARGS = PARSER.parse_args()
    chaining()
