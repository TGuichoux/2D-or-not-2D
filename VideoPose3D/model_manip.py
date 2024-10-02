import numpy as np
import torch
import os
from common.model import TemporalModel
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


if __name__=="__main__":


    filter_widths = [3,3,3,3,3]
    causal = False
    dropout=0.25
    channels=1024
    dense = False
    model_pos = TemporalModel(17, 2, 17,
                            filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                            dense=dense)
    
    checkpoint = "checkpoint"
    resume = False
    evaluate = 'pretrained_h36m_cpn.bin'
    chk_filename = os.path.join(checkpoint, resume if resume else evaluate)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])


    x = -np.load("246fps_2d.npy")
    x = x[:,:,:2,None]
    x = x.reshape(1,246,10,2)

    pelvis = np.array([[0.0, -0.2]])
    r_hip = np.array([[0.15, -0.2]])
    r_knee = np.array([[0.15, -0.4]])
    r_foot = np.array([[0.15, -0.6]])
    l_hip = np.array([[-0.15, -0.2]])
    l_knee = np.array([[-0.15, -0.4]])
    l_foot = np.array([[-0.15, -0.6]])

    # Stack arrays vertically to form a matrix with 2 columns and 7 rows
    additional_kpt = np.vstack((pelvis, r_hip, r_knee, r_foot, l_hip, l_knee, l_foot))
    additional_kpt = np.tile(np.expand_dims(additional_kpt, axis=0), (1, 246, 1, 1))
    print(additional_kpt.shape)

    
    x = torch.from_numpy(np.concatenate((additional_kpt, x), axis=2)).to(torch.float32)
    first_pose2D = x[0,0]

    predicted_3Dpose = -model_pos(x)
    np.save("./246fps_3d", predicted_3Dpose.detach().numpy())

    first_pose = predicted_3Dpose[0,0].detach().numpy()

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.scatter(first_pose[:,0], first_pose[:,1], first_pose[:,2], c=[0,0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3])
    ax.plot([first_pose[7,0], first_pose[8,0]], [first_pose[7,1], first_pose[8,1]], [first_pose[7,2], first_pose[8,2]], color="red")
    ax.plot([first_pose[8,0], first_pose[9,0]], [first_pose[8,1], first_pose[9,1]], [first_pose[8,2], first_pose[9,2]], color="red")
    ax.plot([first_pose[9,0], first_pose[10,0]], [first_pose[9,1], first_pose[10,1]], [first_pose[9,2], first_pose[10,2]], color="red")
    ax.plot([first_pose[0,0], first_pose[7,0]], [first_pose[0,1], first_pose[7,1]], [first_pose[0,2], first_pose[7,2]], color="red")

    ax.plot([first_pose[8,0], first_pose[11,0]], [first_pose[8,1], first_pose[11,1]], [first_pose[8,2], first_pose[11,2]], color="blue")
    ax.plot([first_pose[11,0], first_pose[12,0]], [first_pose[11,1], first_pose[12,1]], [first_pose[11,2], first_pose[12,2]], color="blue")
    ax.plot([first_pose[12,0], first_pose[13,0]], [first_pose[12,1], first_pose[13,1]], [first_pose[12,2], first_pose[13,2]], color="blue")
    
    ax.plot([first_pose[8,0], first_pose[14,0]], [first_pose[8,1], first_pose[14,1]], [first_pose[8,2], first_pose[14,2]], color="green")
    ax.plot([first_pose[14,0], first_pose[15,0]], [first_pose[14,1], first_pose[15,1]], [first_pose[14,2], first_pose[15,2]], color="green")
    ax.plot([first_pose[15,0], first_pose[16,0]], [first_pose[15,1], first_pose[16,1]], [first_pose[15,2], first_pose[16,2]], color="green")

    ax.plot([first_pose[0,0], first_pose[1,0]], [first_pose[0,1], first_pose[1,1]], [first_pose[0,2], first_pose[1,2]], color="purple")
    ax.plot([first_pose[1,0], first_pose[2,0]], [first_pose[1,1], first_pose[2,1]], [first_pose[1,2], first_pose[2,2]], color="purple")
    ax.plot([first_pose[2,0], first_pose[3,0]], [first_pose[2,1], first_pose[3,1]], [first_pose[2,2], first_pose[3,2]], color="purple")

    ax.plot([first_pose[0,0], first_pose[4,0]], [first_pose[0,1], first_pose[4,1]], [first_pose[0,2], first_pose[4,2]], color="cyan")
    ax.plot([first_pose[4,0], first_pose[5,0]], [first_pose[4,1], first_pose[5,1]], [first_pose[4,2], first_pose[5,2]], color="cyan")
    ax.plot([first_pose[5,0], first_pose[6,0]], [first_pose[5,1], first_pose[6,1]], [first_pose[5,2], first_pose[6,2]], color="cyan")
    

    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(first_pose2D[:,0], first_pose2D[:,1])

    plt.savefig("./3Dconversion.png")


