from torch.utils.data import DataLoader
import datetime
import librosa
import lmdb
import logging
import math
import numpy as np
import os
import pickle
import pprint
import pyarrow
import random
import soundfile as sf
import sys
import time
import torch
import torch.nn.functional as F
import configargparse

from DiffGesture.scripts.data_loader.data_preprocessor import DataPreprocessor
from DiffGesture.scripts.data_loader.lmdb_data_loader import SpeechMotionDataset, default_collate_fn
from DiffGesture.scripts.model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from DiffGesture.scripts.model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from DiffGesture.scripts.model.pose_diffusion import PoseDiffusion
from DiffGesture.scripts.utils.average_meter import AverageMeter
from DiffGesture.scripts.utils.data_utils import convert_dir_vec_to_pose, convert_pose_seq_to_dir_vec, resample_pose_seq, dir_vec_pairs, convert_dir_vec_to_pose_2d
from DiffGesture.scripts.utils.train_utils import set_logger, set_random_seed
from utils.utils_VP3D import transform_batch_torch
from DiffGesture.scripts.train_eval.synthesize import generate_gestures
from VideoPose3D.common.model import TemporalModel
from DiffGesture.scripts.model import vocab


import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import subprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

angle_pair = [
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8)
]

change_angle = [0.0034540758933871984, 0.007043459918349981, 0.003493624273687601, 0.007205077446997166]
sigma = 0.1
thres = 0.03
thres_2 = 0.1



def get_speaker_model(net):
    try:
        if hasattr(net, 'module'):
            speaker_model = net.module.z_obj
        else:
            speaker_model = net.z_obj
    except AttributeError:
        speaker_model = None

    if not isinstance(speaker_model, vocab.Vocab):
        speaker_model = None

    return speaker_model


def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))
    if args.model == 'pose_diffusion':
        print("init diffusion model")
        diffusion = PoseDiffusion(args).to(_device)
        diffusion.load_state_dict(checkpoint['state_dict'])

        return args, diffusion, lang_model, speaker_model, pose_dim
    elif args.model == 'multimodal_context':
        print("init multimodal model")
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
        generator.load_state_dict(checkpoint['state_dict'])

        return args, generator, lang_model, speaker_model, pose_dim





def evaluate_testset(test_data_loader, generator, embed_space_evaluator, args, pose_dim, videopose=None, lifting=False, lowering=False, vid=None):

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    # losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    bc = AverageMeter('bc')
    bc_2 = AverageMeter('bc_0.1')
    sym_bc = AverageMeter('sym_bc')
    start = time.time()

    
    xy_index = np.array([i for i in range(int((args.pose_dim/args.dimension)*3)) if i % 3 != 2])

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            #print("testing {}/{}".format(iter_idx, len(test_data_loader)))
            in_text, _, in_text_padded, _, target_vec, in_audio, in_spec, _ = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = in_spec.to(device)
            if args.model == 'multimodal_context':
                speaker_model = get_speaker_model(generator)
                if speaker_model:
                    vid = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                    vid = torch.LongTensor(vid).to(device)
                else:
                    assert False, 'No speaker model'
                    vid = None

            if args.dimension == 3:
                target = target_vec.to(device)  
                pre_seq = target.new_zeros((batch_size, args.n_poses, args.pose_dim + 1))
                pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]

            elif args.dimension == 2:
                if lifting:
                    target = target_vec.to(device)  
                    pre_seq = target.new_zeros((batch_size, args.n_poses, args.pose_dim + 1))
                    pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses, xy_index] # remove z axis
                elif not lifting:
                    target = target_vec[:,:,xy_index].to(device) # remove z axis
                    pre_seq = target.new_zeros((batch_size, args.n_poses, args.pose_dim + 1))
                    pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
                
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            if args.model == 'pose_diffusion':
                out_dir_vec = generator.sample(pose_dim, pre_seq, in_audio)

            elif args.model == 'multimodal_context':
                if args.input_context == 'none':
                    out_dir_vec, *_ = generator(pre_seq, in_text=None, in_audio=None, vid_indices=vid)
                elif args.input_context == 'audio':
                    out_dir_vec, *_ = generator(pre_seq, in_text=None, in_audio=in_audio, vid_indices=vid)
                elif args.input_context == 'text':
                    out_dir_vec, *_ = generator(pre_seq, in_text=in_text_padded, in_audio=None, vid_indices=vid)
                else:
                    out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid)
            if lifting:
                duration = args.n_poses / args.motion_resampling_framerate
                receptive_field = videopose.receptive_field()
                pad = (receptive_field - 1) // 2
                fps = receptive_field / duration
                out_dir_vec = out_dir_vec.reshape(out_dir_vec.shape[0],out_dir_vec.shape[1],-1,2)
                _, res_out_dir_vec = transform_batch_torch(out_dir_vec, duration, fps, pad) # upsample and pad sequence before lifting
                res_out_dir_vec = res_out_dir_vec.reshape(batch_size, -1, args.pose_dim//args.dimension,args.dimension) # (bs, padded upsampled, 9, 2)
                post_out_dir_vec = videopose(res_out_dir_vec)
                out_dir_vec, _ = transform_batch_torch(post_out_dir_vec, duration, args.motion_resampling_framerate, 0) # resample to original framerate
                out_dir_vec = out_dir_vec.reshape(batch_size, args.n_poses, -1)
           
                
            if  args.dimension == 3 or lifting:
                out_dir_vec_bc = out_dir_vec + torch.tensor(args.mean_dir_vec_3d).squeeze(1).unsqueeze(0).unsqueeze(0).cuda()
                beat_vec = out_dir_vec_bc.reshape(target.shape[0], target.shape[1], -1, 3)

                beat_vec = F.normalize(beat_vec, dim = -1)
                all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)
            
            else:
                out_dir_vec_bc = out_dir_vec + torch.tensor(args.mean_dir_vec_2d).squeeze(1).unsqueeze(0).unsqueeze(0).cuda()
                beat_vec = out_dir_vec_bc.reshape(target.shape[0], target.shape[1], -1, 2)

                beat_vec = F.normalize(beat_vec, dim = -1)
                all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 2)

            if lowering: # removing depth axis
                out_dir_vec = out_dir_vec[:,:,xy_index]
                out_dir_vec_bc = out_dir_vec_bc[:,:,xy_index]
                beat_vec = beat_vec[:,:,:,:2]
                all_vec = all_vec[:,:,:2]
                target = target[:,:,xy_index]
              

            # Compute angle difference along temporal axis
            for idx, pair in enumerate(angle_pair):
                vec1 = all_vec[:, pair[0]]
                vec2 = all_vec[:, pair[1]]
                inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                inner_product = torch.clamp(inner_product, -1, 1, out=None)
                angle = torch.acos(inner_product) / math.pi
                angle_time = angle.reshape(batch_size, -1)
                if idx == 0:
                    angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                else:
                    angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
            angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim = -1)
            
            # 
            for b in range(batch_size):
                motion_beat_time = []
                for t in range(2, 33):
                    if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                        if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                            motion_beat_time.append(float(t) / 15.0)
                if (len(motion_beat_time) == 0):
                    continue
                audio = in_audio[b].cpu().numpy()
                audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                sum = 0
                for audio in audio_beat_time:
                    sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                bc.update(sum / len(audio_beat_time), len(audio_beat_time))
                sum_motion = 0
                for motion in motion_beat_time:
                    sum_motion += np.power(math.e, -np.min(np.power((audio_beat_time - motion), 2)) / (2 * sigma * sigma))
                sym_bc.update(sum/len(audio_beat_time) + sum_motion/len(motion_beat_time), len(audio_beat_time))

            for b in range(batch_size):
                motion_beat_time = []
                for t in range(2, 33):
                    if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                        if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres_2 or angle_diff[b][t + 1] - angle_diff[b][t] >= thres_2):
                            motion_beat_time.append(float(t) / 15.0)
                if (len(motion_beat_time) == 0):
                    continue
                audio = in_audio[b].cpu().numpy()
                audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                sum = 0
                for audio in audio_beat_time:
                    sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                bc_2.update(sum / len(audio_beat_time), len(audio_beat_time))
                

            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                #calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                if (args.dimension == 3 or lifting) and not lowering:
                    out_dir_vec += np.array(args.mean_dir_vec_3d).squeeze()
                    out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                else:
                    out_dir_vec += np.array(args.mean_dir_vec_2d).squeeze()
                    out_joint_poses = convert_dir_vec_to_pose_2d(out_dir_vec)
                
                target = target.cpu().numpy()
                if (args.dimension == 3 or lifting) and not lowering:
                    target += np.array(args.mean_dir_vec_3d).squeeze()
                    target_poses = convert_dir_vec_to_pose(target)
                else:
                    target += np.array(args.mean_dir_vec_2d).squeeze()
                    target_poses = convert_dir_vec_to_pose_2d(target)

                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                #accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # print
    ret_dict = {'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        diversity_score = embed_space_evaluator.get_diversity_scores()
        logging.info(
            '[VAL] joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, diversity_score: {:.3f}, BC: {:.3f}, BC 0.07: {:.3f}, Symetric_BC: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                joint_mae.avg, accel.avg, frechet_dist, diversity_score, bc.avg, bc_2.avg, sym_bc.avg, feat_dist, elapsed_time))
        print(
            '[VAL] joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, diversity_score: {:.3f}, BC: {:.3f}, BC 0.07: {:.3f}, Symetric_BC: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                joint_mae.avg, accel.avg, frechet_dist, diversity_score, bc.avg, bc_2.avg, sym_bc.avg, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
        ret_dict['diversity_score'] = diversity_score
        ret_dict['bc'] = bc.avg
        ret_dict['sym_bc'] = sym_bc.avg
        ret_dict['bc_0.1'] = bc_2.avg

    return ret_dict

def evaluate_testset_save_video(test_data_loader, generator, args, lang_model, pose_dim, videopose=None, idx=None, n_samples=1, vid=None):

    n_save = 5

    if args.dimension == 2:
        index = np.array([i for i in range(int(args.pose_dim * 1.5)) if i % 3 != 2])
    with torch.no_grad():
        if idx is not None:
            loader = zip(idx,[test_data_loader.dataset.__getitem__(i) for i in idx])
        else:
            loader = enumerate(test_data_loader,0)
        for iter_idx, data in loader:
            print("testing {}/{}".format(iter_idx, n_save))
            _, _, in_text_padded, target_pose, target_vec, in_audio, in_spec, aux_info = data
            target_3d = target_vec
            if args.dimension == 2:
                # Extract the 2D version of the pose sequence
                target_vec = target_vec[:, :, index]

            # prepare
            select_index = 0
            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            in_spec = in_spec[select_index, :].unsqueeze(0).to(device)
            target_dir_vec = target_vec[select_index, :].unsqueeze(0).to(device)
            target_dir_vec_3d = target_3d[select_index, :].unsqueeze(0).to(device)
            pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1], target_dir_vec.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            if args.model == 'multimodal_context':
                if args.z_type == 'speaker':
                    if not vid:
                        vid = random.randrange(generator.z_obj.n_words)
                    print('vid:', vid)
                    vid = torch.LongTensor([vid]).to(device)
                else:
                    vid = None


            for j in range(n_samples):
                if args.model == 'pose_diffusion':
                    out_dir_vec = generator.sample(pose_dim, pre_seq, in_audio)
                elif args.model == 'multimodal_context':
                    out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid)
                if args.dimension == 2:
                    # Post-processing:
                    duration = args.n_poses / args.motion_resampling_framerate
                    receptive_field = videopose.receptive_field()
                    pad = (receptive_field - 1) // 2
                    fps = receptive_field / duration
                    res_out_dir_vec = resample_pose_seq(out_dir_vec.squeeze(0).cpu().numpy(), duration, fps)
                    res_out_dir_vec = res_out_dir_vec.reshape(-1,args.pose_dim//args.dimension,args.dimension)
                    res_out_dir_vec = np.expand_dims(np.pad(res_out_dir_vec,
                                ((pad, pad), (0, 0), (0, 0)),
                                'edge'), axis=0)
                    res_out_dir_vec = torch.from_numpy(res_out_dir_vec).to(device)
                    post_out_dir_vec = videopose(res_out_dir_vec).cpu().numpy().squeeze(0)
                    post_out_dir_vec = resample_pose_seq(post_out_dir_vec, duration, args.motion_resampling_framerate).reshape(args.n_poses,-1)
                    
                    

                

                audio_npy = np.squeeze(in_audio.cpu().numpy())
                target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
                target_dir_vec_3d = np.squeeze(target_dir_vec_3d.cpu().numpy())
                out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())
                if args.dimension ==  2:
                    post_out_dir_vec = np.squeeze(post_out_dir_vec)
                else:
                    post_out_dir_vec = None

                input_words = []
                for i in range(in_text_padded.shape[1]):
                    word_idx = int(in_text_padded.data[0, i])
                    if word_idx > 0:
                        input_words.append(lang_model.index2word[word_idx])
                sentence = ' '.join(input_words)

                aux_str = '({}, time: {}-{})'.format(
                    aux_info['vid'][0],
                    str(datetime.timedelta(seconds=aux_info['start_time'][0].item())),
                    str(datetime.timedelta(seconds=aux_info['end_time'][0].item())))

                mean_data = np.array(args.mean_dir_vec_2d).reshape(-1, 2)
                mean_data_3d = np.array(args.mean_dir_vec_3d).reshape(-1,3)
                save_path = args.model_save_path
                if not os.path.exists(save_path+f"/short/short-{iter_idx}"):
                        os.makedirs(save_path+f"/short/short-{iter_idx}/")

                if args.dimension == 2:
                    create_video_and_save(
                        save_path+f"/short/short-{iter_idx}", iter_idx, f"render-{iter_idx}-sample-{j}",
                        target_2d=target_dir_vec, output_2d=out_dir_vec, mean_data_2d=mean_data,
                        title="Postprocessed \n"+sentence, audio=audio_npy, aux_str=aux_str, target_3d=target_dir_vec_3d, mean_data_3d=mean_data_3d, output_3d=post_out_dir_vec, sample_idx="-"+str(j))
                else:
                    create_video_and_save(
                        save_path+f"/short/short-{iter_idx}", iter_idx, f"render-{iter_idx}-sample-{j}",
                        target_2d=None, output_2d=None, mean_data_2d=mean_data,
                        title="3D Native \n"+sentence, audio=audio_npy, aux_str=aux_str, target_3d=target_dir_vec_3d, mean_data_3d=mean_data_3d, output_3d=out_dir_vec, sample_idx="-"+str(j))

            if iter_idx>=n_save:
                break

def create_video_and_save(save_path, iter_idx, prefix, target_2d=None, output_2d=None, mean_data_2d=None, title=None,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True, target_3d=None, output_3d=None, mean_data_3d=None, sample_idx=''):
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(16, 16))
   
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3, projection='3d'), fig.add_subplot(2, 2, 4, projection="3d")]
 

    # axes[0].view_init(elev=20, azim=-60)
    # axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data_2d = mean_data_2d.flatten()
    mean_data_3d = mean_data_3d.flatten()
    output_poses_2d = None
    if output_2d is not None:
        output_2d = output_2d + mean_data_2d
        output_poses_2d = convert_dir_vec_to_pose_2d(output_2d)
    target_poses_2d = None
    if target_2d is not None:
        target_2d = target_2d + mean_data_2d
        target_poses_2d = convert_dir_vec_to_pose_2d(target_2d)
    target_poses_3d = None
    if target_3d is not None:
        target_3d = target_3d + mean_data_3d
        target_poses_3d = convert_dir_vec_to_pose(target_3d)
    output_poses_3d = None
    if output_3d is not None:
        output_poses_3d = output_3d + mean_data_3d
        output_poses_3d = convert_dir_vec_to_pose(output_poses_3d)
        
    frame_list_2D = []
    frame_list_3D = []
    def animate(i):
        for k, name in enumerate(['human_2D', 'generated_2D','human_3D','generated_3D']):
            if name == 'human_2D' and target_2d is not None and i < len(target_2d):
                pose = target_poses_2d[i]
            elif name == 'generated_2D' and output_2d is not None and i < len(output_2d):
                pose = output_poses_2d[i]
            elif name == 'human_3D' and target_3d is not None and i < len(target_3d):
                pose = target_poses_3d[i]
            elif name == 'generated_3D' and output_3d is not None and i < len(output_3d):
                pose = output_poses_3d[i]
            else:
                pose = None
            if pose is not None:
                axes[k].clear()
                if pose.shape[-1] == 2:
                    save_fig = plt.figure(figsize=(16,16))
                    ax = save_fig.add_subplot(1,1,1)
                    for j, pair in enumerate(dir_vec_pairs):
                        # Plotting only the x and y coordinates
                        axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                        [pose[pair[0], 1], pose[pair[1], 1]],
                                        linewidth=5)
                        ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                                    [pose[pair[0], 1], pose[pair[1], 1]],
                                    linewidth=10)
                    axes[k].set_xlim(-0.8, 0.8)
                    axes[k].set_ylim(1, -1)
                    axes[k].set_xlabel('x')
                    axes[k].set_ylabel('y')
                    axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(target_2d)))
                        # axes[k].axis('off')
                    if name == 'human_2D':
                        ax.set_xlim(-0.8, 0.8)
                        ax.set_ylim(1, -1)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        save_fig.savefig(f"output/image/human_2D_{i}.png")

                elif pose.shape[-1] == 3:
                    save_fig = plt.figure(figsize=(16,16))
                    ax = save_fig.add_subplot(1,1,1, projection='3d')
                    for j, pair in enumerate(dir_vec_pairs):
                        axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                    [pose[pair[0], 2], pose[pair[1], 2]],
                                    [pose[pair[0], 1], pose[pair[1], 1]],
                                    zdir='z', linewidth=5)
                        ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                                    [pose[pair[0], 2], pose[pair[1], 2]],
                                    [pose[pair[0], 1], pose[pair[1], 1]],
                                    zdir='z', linewidth=10)
                        
                    axes[k].set_xlim3d(-0.5, 0.5)
                    axes[k].set_ylim3d(0.5, -0.5)
                    axes[k].set_zlim3d(0.5, -0.5)
                    axes[k].set_xlabel('x')
                    axes[k].set_ylabel('z')
                    axes[k].set_zlabel('y')
                    axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(target_3d)))
                    # axes[k].axis('off')
                    if name == 'human_3D':
                        ax.set_xlim3d(-0.5, 0.5)
                        ax.set_ylim3d(0.5, -0.5)
                        ax.set_zlim3d(0.5, -0.5)
                        ax.set_xlabel('x')
                        ax.set_ylabel('z')
                        ax.set_zlabel('y')
                        ax.grid(b=None)
                        save_fig.savefig(f"output/image/human_3D_{i}.png")

    if output_2d is not None:
        num_frames = max(len(target_2d), len(output_2d))
    else:
        num_frames = max(len(target_3d), len(output_3d))

    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = '{}/{}.wav'.format(save_path, iter_idx)
        sf.write(audio_path, audio, sr)

    # save video
    try:
        video_path = '{}/temp_{}.mp4'.format(save_path,  iter_idx)
        ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio is not None:
        merged_video_path = '{}/{}.mp4'.format(save_path, prefix)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    if target_3d is not None:
        np.save(save_path+"/gt_3D.npz", target_3d)
    if target_2d is not None:
        np.save(save_path+"/gt_2D.npz", target_2d)
    if output_2d is not None:
        np.save(save_path+"/gen_2D"+sample_idx+".npz", output_2d)
    if output_3d is not None:
        np.save(save_path+"/gen_3D"+sample_idx+".npz", output_3d)
    return output_poses_3d, target_poses_3d, output_poses_2d, target_poses_2d





def main(mode, checkpoint_path, keys=None, lifting=False, lowering=False, model=None, input_context=None):

    args, generator, lang_model, speaker_model, pose_dim = load_checkpoint_and_model(
        checkpoint_path, device)

    
    # random seed
    if args.random_seed >= 0:
        set_random_seed(args.random_seed)

    # set logger
    set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    #logging.info(pprint.pformat(vars(args)))

    # load mean vec
    mean_pose = np.array(args.mean_pose_3d).squeeze()
    mean_dir_vec = np.array(args.mean_dir_vec_3d).squeeze()

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    collate_fn = default_collate_fn

    def load_dataset(path):
        dataset = SpeechMotionDataset(path,
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=speaker_model,
                                      mean_pose=mean_pose,
                                      mean_dir_vec=mean_dir_vec
                                      )
        print(len(dataset))
        return dataset
    if args.dimension ==  2:
        n_joints = args.pose_dim // args.dimension
        videopose = TemporalModel(n_joints, 2, n_joints,
                            filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024,
                            dense=False).eval()
        checkpoint_name = "training_VP3D_logs/200000_subset/vp3D-2024-10-01-10:23/epoch_99-model-24.32-chkpt.pth" #"../VP3D_checkpoints/TED/epoch_12-model-40.96-chkpt.pth" #"../VP3D_checkpoints/TED/epoch_60-model-19.92-chkpt.pth"
        state_dict = torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
        videopose.load_state_dict(state_dict["model_pos"], strict=False)
        videopose.eval()
        videopose.to(device)
        receptive_field = videopose.receptive_field()
        pad = (receptive_field - 1) // 2
    else:
        videopose = None

    val_data_path = args.val_data_path[0]
    if mode == 'eval':
        if (args.dimension == 2 and not lifting) or lowering:
            eval_net_path = 'output/train_h36m_gesture_autoencoder/2D_retrain_h36m_gesture_autoencoder/new_2D_gesture_autoencoder_checkpoint_best.bin'
        else:
            eval_net_path = 'output/train_h36m_gesture_autoencoder/3D_retrain_h36m_gesture_autoencoder/3D-autoencoder_checkpoint_best.bin'
            #eval_net_path = 'output/train_h36m_gesture_autoencoder/gesture_autoencoder_checkpoint_best.bin'

        print("Loaded encoder: ", eval_net_path)
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, eval_net_path, lang_model, device)
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)
        evaluate_testset(data_loader, generator, embed_space_evaluator, args, pose_dim, videopose, lifting, lowering)
    
    elif mode == 'short':
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)
        evaluate_testset_save_video(data_loader, generator, args, lang_model, pose_dim, videopose, n_samples=1)

    elif mode == 'long':
        clip_duration_range = [8,12]

        n_generations = 40
        n_samples = 1

        # load clips and make gestures
        n_saved = 0
        lmdb_env = lmdb.open(val_data_path, readonly=True, lock=False)
        candidate_clips = np.load('candidate_keys.npy')
        keys = candidate_clips[:,0]
        clips_idx = candidate_clips[:,1].astype(np.int32)
        print(keys)
        print(clips_idx)
        paths = []
        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()] if keys is None else keys
            while n_saved < n_generations:  # loop until we get the desired number of results
                # select video
                key = keys[n_saved]#random.choice(keys)

                buf = txn.get(key)
                video = pyarrow.deserialize(buf)
                vid = video['vid']
                clips = video['clips']

                # select clip
                n_clips = len(clips)
                if n_clips == 0:
                    continue
                clip_idx = random.randrange(n_clips) if clips_idx is None else clips_idx[n_saved]

                clip_poses = clips[clip_idx]['skeletons_3d']
                clip_audio = clips[clip_idx]['audio_raw']
                clip_words = clips[clip_idx]['words']
                clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]


                clip_poses = resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                                args.motion_resampling_framerate)
                target_dir_vec = convert_pose_seq_to_dir_vec(clip_poses)
                target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
                target_dir_vec -= np.array(args.mean_dir_vec_3d).squeeze()

                target_dir_vec_3d = target_dir_vec
                if args.dimension == 2:
                    target_dir_vec_2d = target_dir_vec.reshape(target_dir_vec.shape[0],-1,3)[...,:2].reshape(target_dir_vec.shape[0],-1)
                else:
                    target_dir_vec_2d = None

                # check duration
                clip_duration = clip_time[1] - clip_time[0]
                if clip_duration < clip_duration_range[0] or clip_duration > clip_duration_range[1]:
                    continue

                #print(key, clip_idx)
                # synthesize
                for selected_vi in range(len(clip_words)):  # make start time of input text zero
                    clip_words[selected_vi][1] -= clip_time[0]  # start time
                    clip_words[selected_vi][2] -= clip_time[0]  # end time
                
                for i in range(n_samples):
                    vid_idx = random.sample(range(0, speaker_model.n_words), 1)[0]
                    if args.model == 'pose_diffusion':
                        if args.dimension == 2:
                            out_dir_vec_2d = generate_gestures(args=args, generator=generator, lang_model=lang_model, audio=clip_audio, words=clip_words, pose_dim=pose_dim, 
                                                        seed_seq=target_dir_vec_2d[0:args.n_pre_poses], fade_out=False, device=device)
                        else:
                            out_dir_vec_3d = generate_gestures(args=args, generator=generator, lang_model=lang_model, audio=clip_audio, words=clip_words, pose_dim=pose_dim, 
                                                        seed_seq=target_dir_vec_3d[0:args.n_pre_poses], fade_out=False, device=device)
                                              
                            out_dir_vec_2d = None
                        
                    elif args.model == 'multimodal_context':
                        
                        if args.dimension == 2:
                            out_dir_vec_2d = generate_gestures(args=args, generator=generator, lang_model=lang_model, audio=clip_audio, words=clip_words, pose_dim=pose_dim, 
                                                        seed_seq=target_dir_vec_2d[0:args.n_pre_poses], fade_out=False, vid=vid_idx, device=device)
                        else:
                            out_dir_vec_3d = generate_gestures(args=args, generator=generator, lang_model=lang_model, audio=clip_audio, words=clip_words, pose_dim=pose_dim, 
                                                        seed_seq=target_dir_vec_3d[0:args.n_pre_poses], fade_out=False, vid=vid_idx, device=device)
                            out_dir_vec_2d = None

                    if args.dimension == 2:
                        if lifting:
                            out_dir_vec_2d = resample_pose_seq(out_dir_vec_2d, clip_time[1]-clip_time[0], args.motion_resampling_framerate)
                            post_out_dir_vec = resample_pose_seq(out_dir_vec_2d, clip_time[1]-clip_time[0], 107)
                            post_out_dir_vec = post_out_dir_vec.reshape(-1,args.pose_dim//args.dimension,args.dimension)
                            post_out_dir_vec = np.expand_dims(np.pad(post_out_dir_vec,
                                        ((pad, pad), (0, 0), (0, 0)),
                                        'edge'), axis=0)
                            post_out_dir_vec = torch.from_numpy(post_out_dir_vec).to(device)
                            post_out_dir_vec = videopose(post_out_dir_vec).detach().cpu().numpy().squeeze(0)
                            post_out_dir_vec = post_out_dir_vec.reshape(post_out_dir_vec.shape[0],-1)
                            out_dir_vec_3d = resample_pose_seq(post_out_dir_vec, clip_time[1] - clip_time[0], args.motion_resampling_framerate)
                        else:
                            out_dir_vec_3d = None
                       
                    if args.dimension == 3 and lowering:
                        out_dir_vec_2d = out_dir_vec_3d.reshape(out_dir_vec_3d.shape[0], -1,3)[:,:,:2].reshape(out_dir_vec_3d.shape[0],-1)
                        
                        
                    
                    # make a video
                    aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                        str(datetime.timedelta(seconds=clip_time[1])))
                    save_path = args.model_save_path
                    if not os.path.exists(save_path+f"/long/"):
                        os.makedirs(save_path+f"/long/")


                    if lowering:
                        title = f"Generated samples with {args.model}, \n 2D gestures obtained by lowering 3D gestures"
                        postpro = 'lowering'
                    elif lifting:
                        title = f"Generated samples with {args.model}, \n 3D gestures obtained by lifting 2D gestures"
                        postpro = 'lifting'
                    elif not lifting and not lowering:
                        postpro = 'normal'
                        title = f"Generated samples with {args.model}, \n No lifting, no lowering"
                  

                    sentence = ''
                    for w in  clip_words:
                        sentence+=w[0]+' '
                    sentence = sentence[:-1]
                    out_dir_vec_3d = out_dir_vec_3d[:target_dir_vec_3d.shape[0]]
                    out_dir_vec_2d = out_dir_vec_2d[:target_dir_vec_2d.shape[0]] if out_dir_vec_2d is not None else None
          
                    # create_video_and_save(
                    #     save_path+f"/long", n_saved, f'{postpro}-{model}-{args.dimension}-{args.input_context}-{key}-{i}',
                    #     target_dir_vec_2d, out_dir_vec_2d, np.array(args.mean_dir_vec_2d).reshape(-1, 2),
                    #     title=title, audio=clip_audio, aux_str=aux_str, target_3d=target_dir_vec_3d, mean_data_3d = np.array(args.mean_dir_vec_3d).reshape(-1,3), output_3d=out_dir_vec_3d, sample_idx="-"+str(i))


                    
                    if args.dimension == 3 or lifting:
                        out_dir_vec_3d = out_dir_vec_3d + np.array(args.mean_dir_vec_3d).squeeze()
                        out_poses = convert_dir_vec_to_pose(out_dir_vec_3d)
                        pickle_save_path = f'output/saved_pickles/{model}/'
                        save_dict = {
                            'sentence': sentence, 'audio': clip_audio.astype(np.float32),  
                            'out_dir_vec': out_dir_vec_3d, 'out_poses': out_poses,
                            'aux_info': '{}_{}_{}'.format(vid, vid_idx, clip_idx),
                            'human_dir_vec': target_dir_vec_3d + mean_dir_vec,
                        }
                       
                        assert False, "To change"
                        with open(pickle_save_path+f'normal-human-{args.dimension}-{key}-{clip_idx}.pkl', 'wb') as f:
                            pickle.dump(save_dict, f)

                        paths.append(pickle_save_path+f'normal-human-{args.dimension}-{key}-{clip_idx}.pkl')
                        assert len(clip_audio.shape) == 1  # 1-channel, raw signal
                        audio = clip_audio.astype(np.float32)
                        sr = 16000
                        audio_path = pickle_save_path+f'normal-human-{args.dimension}-{key}-{clip_idx}.wav'
                        sf.write(audio_path, audio, sr)

                n_saved += 1
   
    else:
        assert False, 'wrong mode'





if __name__ == '__main__':
 
    parser = configargparse.ArgParser()
    parser.add("--mode", type=str)
    parser.add("--postprocessing", type=str, help="lifting, lowering")
    parser.add("--dimension", type=int)
    parser.add("--context", type=str)
    parser.add("--model", type=str, help="multimodal_context or pose_diffusion", default='pose_diffusion')
    _args = parser.parse_args()


    print(f"Mode:  {_args.mode}")
    print(f"Postprocessing: {_args.postprocessing}")
    print(f"Dimension: {_args.dimension}")
    print(f"Model: {_args.model}")

    if _args.dimension ==  2:
        if _args.model == 'pose_diffusion':
            
            if _args.context == 'audio':
                ckpt_path = 'output/train_diffusion_ted/diffusion_2D-checkpoints/diffgesture-2D_checkpoint_499.bin'
            elif _args.context == 'none':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''
            else:
                assert False, 'wrong context for pose diffusion'
        elif _args.model == 'multimodal_context':
            if _args.context == 'both':
                ckpt_path = 'output/train_multimodal_context/2D_multimodal-checkpoint/trimodal-2D-both/trimodal-2D-both_checkpoint_best.bin'
            elif _args.context == 'audio':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''
            elif _args.context == 'text':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''            
            elif _args.context == 'none':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''

    else:
        if _args.model == 'pose_diffusion':
            if _args.context == 'audio':
                ckpt_path = 'output/train_diffusion_ted/diffusion_3D-checkpoints/diffgesture-3D_checkpoint_499.bin'
            elif _args.context == 'none':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''
            else:
                assert False, 'wrong context for pose diffusion'
        elif _args.model == 'multimodal_context':
            if _args.context == 'both':
                ckpt_path = 'output/train_multimodal_context/3D_multimodal-checkpoint/trimodal-3D-both/trimodal-3D-both_checkpoint_best.bin'
            elif _args.context == 'audio':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''
            elif _args.context == 'text':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''            
            elif _args.context == 'none':
                raise NotImplementedError('Train model and save checkpoint')
                ckpt_path = ''
           
    print("Evaluated model: ", ckpt_path)

    main(_args.mode, ckpt_path, lifting=(_args.postprocessing=='lifting'), lowering=(_args.postprocessing=='lowering'), model=_args.model, input_context=_args.context)
