from TED3D.src.scripts.data_loader.lmdb_data_loader import *
# Standard library imports
#from time import time
import time
# Third-party library imports
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm

# Local TED3D library specific imports
from TED3D.src.scripts.data_loader.lmdb_data_loader import *
from TED3D.src.scripts.utils.data_utils import convert_dir_vec_to_pose, resample_pose_seq, convert_pose_seq_to_dir_vec
from TED3D.src.scripts.utils.vocab_utils import build_vocab

# Local VideoPose3D library specific imports
from VideoPose3D.common.camera import (camera_to_world, image_coordinates,
                                       normalize_screen_coordinates,
                                       world_to_camera)
from VideoPose3D.common.generators import UnchunkedGenerator
from VideoPose3D.common.h36m_dataset import Human36mDataset
from VideoPose3D.common.loss import mpjpe
from VideoPose3D.common.model import TemporalModel
from VideoPose3D.common.skeleton import Skeleton
from VideoPose3D.common.utils import deterministic_random
from VideoPose3D.common.visualization import render_animation

# Local project-specific utility imports
#from utils_VP3D import evaluate, transform_upscale, upscale, resample_pose_seq, pad_tensor #convert_pose_seq_to_dir_vec
#from visu_utils import convert_to_gif
import omegaconf
import json
import yaml

from utils.utils_VP3D import transform_batch_torch, transform_sequence_torch


if __name__=="__main__":
    torch.cuda.empty_cache()
    with open(f"./config/train_VP3D_TED.yml", "r") as f:
        config = omegaconf.OmegaConf.create(yaml.safe_load(f))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    
    skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    skeleton._parents[11] = 8
    skeleton._parents[14] = 8

    skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'), (1, 5, 'darkgreen'),
                       (5, 6, 'limegreen'), (6, 7, 'darkseagreen')]
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                    (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length

    mean_dir_vec =np.array([ 0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916])
    mean_pose = np.array([ 0.0000306,  0.0004946,  0.0008437,  0.0033759, -0.2051629, -0.0143453,  0.0031566, -0.3054764,  0.0411491,  0.0029072, -0.4254303, -0.001311 , -0.1458413, -0.1505532, -0.0138192, -0.2835603,  0.0670333,  0.0107002, -0.2280813,  0.112117 , 0.2087789,  0.1523502, -0.1521499, -0.0161503,  0.291909 , 0.0644232,  0.0040145,  0.2452035,  0.1115339,  0.2051307])


    



    train_dataset = SpeechMotionDataset(config.root_data_dir+config.train_lmdb_path,
                                        n_poses=config.n_poses,
                                        subdivision_stride=10, #10
                                        pose_resampling_fps=config.pose_resampling_fps, #15
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=mean_pose,
                                        remove_word_timing=('both' == 'text')
                                        )
    

    test_dataset = SpeechMotionDataset(config.root_data_dir+config.val_lmdb_path,
                                        n_poses=config.n_poses,
                                        subdivision_stride=10, #10
                                        pose_resampling_fps=config.pose_resampling_fps, #15
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=mean_pose,
                                        remove_word_timing=('both' == 'text')
                                        )
    lang_model = build_vocab('words', [train_dataset, test_dataset], config.root_data_dir+config.vocab_cache_path, config.root_data_dir+config.wordembed_path,
                                300)
    test_dataset.set_lang_model(lang_model)
    train_dataset.set_lang_model(lang_model)

    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0,config.train_subset_size))
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0,config.val_subset_size))

    valloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=default_collate_fn)
    trainloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=default_collate_fn)

    ##### Init model 

    filter_widths = config.filter_widths
    causal = config.causal
    dropout = config.dropout
    channels = config.channels
    dense = config.dense
    N_joints = config.n_joints
    model_pos = TemporalModel(N_joints, 2, N_joints,
                            filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                            dense=dense)
    model_pos.train()
    model_pos.to(device)

    causal_shift = False
    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side    
        
    duration = config.n_poses / config.pose_resampling_fps
    upsample_fps = receptive_field / duration

    #### Training ####
    chk_path = config.checkpoint+"/"+str(config.train_subset_size)+f'_subset/'+config.name+"-"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    print('Saving checkpoint to', chk_path)
    print(f"Training begins at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}")
    with open(chk_path+"/config.json", "w") as f:
        json.dump(omegaconf.OmegaConf.to_container(config, resolve=True), f, indent=4)
    lr = config.lr
        
    lr_decay = config.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = config.initial_momentum 
    final_momentum = config.final_momentum

    optimizer = optim.Adam(model_pos.parameters(), lr=lr, amsgrad=True)


    epoch=0
    for epoch in tqdm(range(config.epochs)):
        start_time = time.time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos.train()

        for _, _, _, batch_3d, target_vec, _, _, _ in tqdm(trainloader, leave=False, total=len(trainloader)):
            batch_size, n_poses, _ = target_vec.shape 
            target_vec = target_vec.to(device)
            target_3d, inputs_3d = transform_batch_torch(target_vec.reshape(batch_size, n_poses, -1, 3), duration, upsample_fps, pad)
            target_3d = target_3d.squeeze(1)
            inputs_3d = inputs_3d.squeeze(1)
            inputs_2d = inputs_3d[...,:2].contiguous()
            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, target_3d)
            epoch_loss_3d_train += target_3d.shape[0]*target_3d.shape[1] * loss_3d_pos.item()
            N += target_3d.shape[0]*target_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
        with torch.no_grad():
            model_pos.load_state_dict(model_pos.state_dict())
            model_pos.eval()


            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            
            if not config.no_eval:
                # Evaluate on test set
                for in_text, text_lengths, in_text_padded, batch_3d, target_vec, in_audio, in_spec, aux_info in valloader:
                    batch_size, n_poses, _ = target_vec.shape 
                    target_vec = target_vec.to(device)
                    target_3d, inputs_3d =  transform_batch_torch(target_vec.reshape(batch_size, n_poses, -1, 3), duration, upsample_fps, pad)
                    target_3d = target_3d.squeeze(1)
                    inputs_3d = inputs_3d.squeeze(1)
                    inputs_2d = inputs_3d[...,:2].contiguous()

                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, target_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

            # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for in_text, text_lengths, in_text_padded, batch_3d, target_vec, in_audio, in_spec, aux_info in trainloader:
                    batch_size, n_poses, _ = target_vec.shape 
                    target_vec = target_vec.to(device)
                    target_3d, inputs_3d =  transform_batch_torch(target_vec.reshape(batch_size, n_poses, -1, 3), duration, upsample_fps, pad)
                    target_3d = target_3d.squeeze(1)
                    inputs_3d = inputs_3d.squeeze(1)
                    inputs_2d = inputs_3d[...,:2].contiguous()

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, target_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    
                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
                
        elapsed = (time.time() - start_time)/60

        if config.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1]  *1000))
        
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch/config.epochs * np.log(initial_momentum/final_momentum))
        model_pos.set_bn_momentum(momentum)
        


        if epoch % config.checkpoint_frequency == 0:
            
            save_path = chk_path+f"/epoch_{epoch}"
            torch.save({
                'loss_train': torch.tensor(losses_3d_train[-1] * 1000),
                'loss_train_eval': torch.tensor(losses_3d_train_eval[-1] * 1000),
                'loss_valid': torch.tensor(losses_3d_valid[-1] * 1000),
            }, save_path+"-loss.pth")
            torch.save({    
                'lr': lr,  
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict()}, save_path+"-model-"+str(round(losses_3d_train[-1]*1000, 2))+"-chkpt.pth")
    