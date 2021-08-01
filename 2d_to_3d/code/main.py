import h5py
import numpy as np
import sys
import supplemental_code
import matplotlib.pyplot as plt
import argparse
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchgeometry as tgm
from tqdm import tqdm
import math

def morphable_model(bfm, alpha, delta):
    color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color = np.reshape(color, (-1, 3))
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.float32).T

    # Identity weigths
    id_mu = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    id_mu = np.reshape(id_mu, (-1, 3))
    id_mu = torch.tensor(id_mu, dtype=torch.float32)
    id_basis = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    id_basis = np.reshape(id_basis, (-1, 3, 199))[:,:,:30]
    id_basis = torch.tensor(id_basis, dtype=torch.float32)
    id_var = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)[:30]
    id_var = torch.tensor(id_var, dtype=torch.float32)
    id = id_mu + torch.matmul(id_basis, torch.sqrt(id_var)*alpha)

    # Expression weigths
    exp_mu = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    exp_mu = np.reshape(exp_mu, (-1, 3))
    exp_mu = torch.tensor(exp_mu, dtype=torch.float32)
    exp_basis = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    exp_basis = np. reshape(exp_basis, (-1, 3, 100))[:,:,:20]
    exp_basis = torch.tensor(exp_basis, dtype=torch.float32)
    exp_var = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)[:20]
    exp_var = torch.tensor(exp_var, dtype=torch.float32)
    exp = exp_mu + torch.matmul(exp_basis, torch.sqrt(exp_var)* delta)

    face = id + exp

    return(face, color, triangles)

def pinhole(face, omega, tau):
    # Construct transformation matrix
    tau = tau.unsqueeze(dim=-1)
    omega = tgm.deg2rad(omega) # Convert degrees to radian values
    omega_x = omega[0].unsqueeze(dim=-1)
    omega_y = omega[1].unsqueeze(dim=-1)
    omega_z = omega[2].unsqueeze(dim=-1)
    zeros = torch.zeros(1)
    ones = torch.ones(1)

    Rx = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, torch.cos(omega_x), -torch.sin(omega_x)]),
                torch.stack([zeros, torch.sin(omega_x), torch.cos(omega_x)])
                ]).reshape(3,3)

    Ry = torch.stack([
                torch.stack([torch.cos(omega_y), zeros, torch.sin(omega_y)]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([-torch.sin(omega_y), zeros, torch.cos(omega_y)])
                ]).reshape(3,3)

    Rz = torch.stack([
                torch.stack([torch.cos(omega_z), -torch.sin(omega_z), zeros]),
                torch.stack([torch.sin(omega_z), torch.cos(omega_z), zeros]),
                torch.stack([zeros, zeros, ones])
                ]).reshape(3,3)

    R = torch.mm(Rz, torch.mm(Ry, Rx))
    T = torch.cat([R, tau],  dim=1)
    T = torch.cat([T, torch.stack([zeros, zeros, zeros, ones],  dim=1)])

    # Construct Pprojection and viewpoint matrix
    face_np = face.data.numpy()
    face_np = np.hstack((face_np, np.ones((face_np.shape[0],1))))
    l = np.min(face_np[:,0])
    r = np.max(face_np[:,0])
    b = np.min(face_np[:,1])
    t = np.max(face_np[:,1])
    n = np.min(face_np[:,2])
    f = np.max(face_np[:,2])

    P = np.array([[(2*n)/(r-l), 0, (r+l)/(r-l), 0],
                    [0, (2*n)/(t-b), (t+b)/(t-b), 0],
                    [0, 0, -((f+n)/(f-n)), -((2*f*n)/(f-n))],
                    [0, 0, -1, 0]])

    V = np.array([[(r-l)/2, 0, 0, (r+l)/2],
                    [0, (t-b)/2, 0, (t+b)/2],
                    [0, 0, 1/2, 1/2],
                    [0, 0, 0, 1]])

    # Map 3D face to 2D face
    pi = V.dot(P)
    pi = torch.tensor(pi, dtype=torch.float32)
    ones = torch.ones((face.shape[0],1))
    face = torch.cat([face, ones], dim=1)
    face = torch.mm(T, face.T)
    face_2D = torch.mm(pi, face).T[:,:3]

    return(face_2D)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.omega = torch.tensor(ARGS.omega, dtype=torch.float32)
        self.omega = torch.nn.Parameter(self.omega, requires_grad=True)
        self.tau = torch.tensor(ARGS.tau, dtype=torch.float32)
        self.tau = torch.nn.Parameter(self.tau, requires_grad=True)
        self.alpha = np.random.uniform(ARGS.alpha_range[0], ARGS.alpha_range[1], 30)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.alpha = torch.nn.Parameter(self.alpha, requires_grad=True)
        self.delta = np.random.uniform(ARGS.delta_range[0], ARGS.delta_range[1], 20)
        self.delta = torch.tensor(self.delta, dtype=torch.float32)
        self.delta = torch.nn.Parameter(self.delta, requires_grad=True)

    def forward(self, bfm):
        face_3D, color, triangles = morphable_model(bfm, self.alpha, self.delta)
        face_2D = pinhole(face_3D, self.omega, self.tau)
        pred = get_landmarks_pred(face_2D)

        return(pred)

def get_landmarks_pred(face):
    f = open('../data/Landmarks68_model2017-1_face12_nomouth.anl','r')
    lines = f.readlines()
    inds = []
    for line in lines: inds.append(int(line))
    pred = face[inds][:,:2]
    return(pred)

def get_landmarks_gt(img, pred):
    gt_original = supplemental_code.detect_landmark(img)
    gt = np.copy(gt_original)
    gt[:,1] = gt[:,1]*-1 #flip horizontally
    gt_mean = np.mean(gt, axis=0)
    gt = gt - gt_mean #center around origin
    gt = torch.tensor(gt, dtype=torch.float32)
    range_pred = torch.max(pred, dim=0)[0][0].item() - torch.min(pred, dim=0)[0][0].item()
    range_gt = torch.max(gt, dim=0)[0][0].item() - torch.min(gt, dim=0)[0][0].item()
    ratio = range_pred/range_gt
    gt = gt*ratio

    return(gt, gt_original.T)

def plot_landmarks(gt, pred, name):
    gt  = gt.data.numpy()
    pred  = pred.data.numpy()
    plt.scatter(gt[:,0], gt[:,1], s = 15, alpha=0.8, c='tab:red', label='ground truth')
    plt.scatter(pred[:,0], pred[:,1], s = 15, alpha=0.8, c='tab:blue', label='prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('../tmp/landmarks_{}.png'.format(name))
    plt.clf()


def save_face_model(model, bfm, name):
    face_3D, color, triangles = morphable_model(bfm, model.alpha, model.delta)
    face_2D = pinhole(face_3D, model.omega, model.tau)
    supplemental_code.save_obj('../tmp/2D_{}.obj'.format(name), face_2D, color, triangles)
    supplemental_code.save_obj('../tmp/3D_{}.obj'.format(name), face_3D, color, triangles)

def train_model(model, optimizer, bfm, gt, img):
    losses = []
    for iter in tqdm(range(ARGS.max_iters)):
        pred = model(bfm)
        loss = compute_loss(gt, pred, model.alpha, model.delta)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (iter > ARGS.patience*2):
            old = int(np.mean(losses[-ARGS.patience*2:-ARGS.patience]))
            new = int(np.mean(losses[-ARGS.patience:]))
            if old <= new:
                break

    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('../tmp/loss.png')
    plt.clf()

    return(model, losses)

def compute_loss(gt, pred, alpha, delta):
    l_lan = torch.mean((pred-gt).pow(2)[:,:])
    l_reg = ARGS.lambda_alpha*torch.sum(alpha.pow(2)) + ARGS.lambda_delta*torch.sum(delta.pow(2))
    l_fit = l_lan + l_reg

    return(l_fit)

def interpolate(x, y, pixels):
    (x1, y1, q11), (x1, y2, q12), (x2, y1, q21), (x2, y2, q22) = sorted(pixels)
    c = (q11*(x2-x)*(y2-y) + q21*(x-x1)*(y2-y) + q12*(x2-x)*(y-y1) + q22*(x-x1)*(y-y1)) / ((x2-x1)*(y2-y1))

    return(c)

def texturize(model, bfm, img, gt, name):
    # Transform prediction to match ground truth image
    face_3D, color, triangles = morphable_model(bfm, model.alpha, model.delta)
    face_2D = pinhole(face_3D, model.omega, model.tau)
    face_2D[:,1] = face_2D[:,1]*-1 #flip horizontally

    pred = get_landmarks_pred(face_2D).data.T
    pred = face_2D.data.numpy()[:,:2].T
    range_pred = np.max(pred[0]) - np.min(pred[0])
    range_gt =  np.max(gt[0]) - np.min(gt[0])
    ratio = range_pred/range_gt
    pred = pred/ratio + np.mean(gt, axis=1).reshape((2,1))

    plt.scatter(pred[0], pred[1], s = 15, alpha=0.8, c='tab:blue', label='prediction')
    plt.scatter(gt[0], gt[1], s = 15, alpha=0.8, c='tab:red', label='ground truth')
    plt.legend()
    plt.grid(True)
    plt.savefig('../tmp/texturize_{}.png'.format(name))

    # Infer locations from original image and infer color using bilinear interpolation
    color = []
    for p in pred.T:
        p_color = []
        for i in range(3):
            p1 = math.floor(p[1])
            p0 = math.floor(p[0])
            img_c = []
            img_c.append((p1, p0, img[p1][p0][i]))
            img_c.append((p1+1, p0+1, img[p1+1][p0+1][i]))
            img_c.append((p1+1, p0, img[p1+1][p0][i]))
            img_c.append((p1, p0+1, img[p1][p0+1][i]))
            c = interpolate(p[1], p[0], img_c)
            p_color.append(int(c))
        color.append(p_color)
    color_RGB =  np.stack(color, axis=0)/255
    color = color_RGB[...,::-1]

    face_2D[:,1] = face_2D[:,1]*-1 #flip horizontally
    supplemental_code.save_obj('../tmp/2D_texture_{}.obj'.format(name), face_2D, color, triangles)
    supplemental_code.save_obj('../tmp/3D_texture_{}.obj'.format(name), face_3D, color, triangles)



def main():
    bfm = h5py.File('../data/model2017-1_face12_nomouth.h5', 'r')

    if ARGS.video == False:
        model = Net()
        optimizer = torch.optim.Adam([{'params': model.omega, 'lr': 1},
                                        {'params': model.tau, 'lr': 1},
                                        {'params': model.alpha, 'lr': 0.1},
                                        {'params': model.delta, 'lr': 0.1}])

        # Obtain ground truth landmarks
        img = cv2.imread('../data/faces/img_1.png')
        gt, gt_original = get_landmarks_gt(img, model(bfm))

        # Train network and create textured 3D model
        save_face_model(model, bfm, 'start')
        model, losses = train_model(model, optimizer, bfm, gt, img)
        save_face_model(model, bfm, 'stop')
        texturize(model, bfm, img, gt_original, 'img')


    else:
        print('FRAME 1')
        model = Net()
        optimizer = torch.optim.Adam([  {'params': model.omega, 'lr': 1},
                                        {'params': model.tau, 'lr': 1},
                                        {'params': model.alpha, 'lr': 0.1},
                                        {'params': model.delta, 'lr': 0.1}])

        # Obtain ground truth landmarks
        frame = cv2.imread('../data/faces/frame_0.png')
        gt, gt_original = get_landmarks_gt(frame, model(bfm))

        # Train network and create textured 3D model
        model, losses = train_model(model, optimizer, bfm, gt, frame)
        save_face_model(model, bfm, 'frame_0')
        texturize(model, bfm, frame, gt_original, 'frame_0')

        for f in range(2,4):
            print('FRAME {}'.format(f))
            model = Net()
            optimizer = torch.optim.Adam([  {'params': model.omega, 'lr': 1},
                                            {'params': model.tau, 'lr': 1},
                                            {'params': model.delta, 'lr': 0.1}])

            # Obtain ground truth landmarks
            frame = cv2.imread('../data/faces/frame_{}.png'.format(f))
            gt, gt_original = get_landmarks_gt(frame, model(bfm))

            # Train network and create textured 3D model
            model, losses = train_model(model, optimizer, bfm, gt, frame)
            save_face_model(model, bfm, 'frame_{}'.format(f))
            texturize(model, bfm, frame, gt_original, 'frame_{}'.format(f))



if __name__ == "__main__":
    plt.style.use('seaborn')
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--omega', default=[0, 0, 0], type=list,
                        help='initial viewpoint rotation')
    PARSER.add_argument('--tau', default=[0, 0, 0], type=list,
                        help='initial viewpoint translation')
    PARSER.add_argument('--alpha_range', default=[-1,1], type=str,
                        help='range for sampling alpha')
    PARSER.add_argument('--delta_range', default=[-1,1], type=str,
                        help='range for sampling delta')
    PARSER.add_argument('--lambda_alpha', default=1, type=float,
                        help='regularization weight for alpha')
    PARSER.add_argument('--lambda_delta', default=1, type=float,
                        help='regularization weight for delta')
    PARSER.add_argument('--max_iters', default=100, type=int,
                        help='maximum number of iterations')
    PARSER.add_argument('--patience', default=10, type=int,
                        help='number of losses considered for early stopper')
    PARSER.add_argument('--video', default=False, type=int,
                        help='learn a sequence of frames')

    ARGS = PARSER.parse_args()

    main()
