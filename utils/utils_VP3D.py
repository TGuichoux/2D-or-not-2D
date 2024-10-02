from TED3D.src.scripts.utils.data_utils import convert_dir_vec_to_pose, resample_pose_seq
import numpy as np
import torch
import torch.nn.functional as F
import re

import librosa
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize


##### Torch version ####

def transform_sequence_torch(seq, duration, fps, pad):
    device = seq.device
    seq = torch.tensor(seq, dtype=torch.float32, device=device)
    res_dir_vec = resample_pose_seq_torch(seq, duration, fps).squeeze()
    res_dir_vec = seq.squeeze()
    res_dir_vec = res_dir_vec.view(res_dir_vec.shape[0], -1, 3)

    pad_tuple = (0, 0, 0, 0, pad, pad)
    padded_res_dir_vec = torch.nn.functional.pad(res_dir_vec.unsqueeze(0), pad_tuple, 'replicate', 0)
    return padded_res_dir_vec.to(torch.float32), res_dir_vec.to(torch.float32)

def transform_batch_torch(batched_sequence, duration, fps, pad):
    new_batch_padded = []
    new_batch = []
    for seq in batched_sequence:
        padded_seq, unpadded_seq = transform_sequence_torch(seq, duration, fps, pad)
        new_batch.append(unpadded_seq)
        new_batch_padded.append(padded_seq)
    return torch.stack(new_batch).to(torch.float32), torch.stack(new_batch_padded).to(torch.float32)


def linear_interpolate(x, y, x_new):
    """
    Perform linear interpolation on a set of points for 3D data.

    Args:
    x (Tensor): Original x-coordinates (1D).
    y (Tensor): Original y-coordinates (values) (3D).
    x_new (Tensor): New x-coordinates for interpolation (1D).

    Returns:
    Tensor: Interpolated y-values corresponding to x_new (3D).
    """
    n = x.shape[0]
    # Prepare indices for gathering
    x_new = x_new.unsqueeze(1).expand(-1, y.size(1)  * y.size(2))
    idx = torch.searchsorted(x, x_new, right=True).clamp(1, len(x)-1)


    # Gather values around each new x coordinate
    x0 = x[idx - 1]#.view(-1, 1)
    x1 = x[idx]#.view(-1, 1)
    y0 = torch.gather(y.view(n, -1), 0, (idx - 1)).view(-1, y.size(1)* y.size(2))
    y1 = torch.gather(y.view(n, -1), 0, idx).view(-1, y.size(1)* y.size(2))

    # Linear interpolation formula
    return (y0 + (x_new - x0) * (y1 - y0) / (x1 - x0)).reshape(-1,y.shape[1], y.shape[2]).to(torch.float32)

def resample_pose_seq_torch(poses, duration_in_sec, fps ):
    n = poses.shape[0]
    x = torch.arange(0, n, dtype=torch.float64,device=poses.device)
    y = poses
    expected_n = duration_in_sec * fps
    x_new = torch.arange(0, n, n / expected_n,device=poses.device)
    interpolated_y = linear_interpolate(x, y, x_new)

    return interpolated_y.to(torch.float32)


##### Numpy version #####

def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    return interpolated_y.astype(np.float32)


def transform_sequence(seq, duration, fps, pad):
    res_dir_vec = resample_pose_seq(seq, duration, fps).squeeze()
    res_dir_vec = res_dir_vec.reshape(res_dir_vec.shape[0],-1,3)

    return np.expand_dims(np.pad(res_dir_vec,((pad, pad), (0, 0), (0, 0)), 'edge'), axis=0).astype(np.float32), res_dir_vec.astype(np.float32)

def transform_batch(batched_sequence, duration, fps, pad):
    new_batch_padded = []
    new_batch = []
    for seq in batched_sequence:
        padded_seq, unpadded_seq = transform_sequence(seq, duration, fps, pad)
        new_batch.append(unpadded_seq.astype(np.float32))
        new_batch_padded.append(padded_seq.astype(np.float32))
    return np.array(new_batch).astype(np.float32), np.array(new_batch_padded).astype(np.float32)






if __name__ == "__main__":
    eps = 1e-5
    def test_resample_pose_seq():
        duration_in_sec = 10
        fps = 30
        poses_np = np.random.rand(34, 9, 3) 
        poses_torch = torch.tensor(poses_np, dtype=torch.float32)

        result_np = resample_pose_seq(poses_np, duration_in_sec, fps).astype(np.float32)
        result_torch = resample_pose_seq_torch(poses_torch, duration_in_sec, fps).cpu().numpy().astype(np.float32)
        assert np.allclose(result_np, result_torch, atol=eps), "resample_pose_seq functions are not equivalent"

    def test_transform_sequence():
        duration = 10
        fps = 30
        pad = 5
        seq_np = np.random.rand(34, 9, 3) 
        seq_torch = torch.tensor(seq_np, dtype=torch.float32)

        result_np = transform_sequence(seq_np, duration, fps, pad)
        result_torch = transform_sequence_torch(seq_torch, duration, fps, pad)
        result_torch = (result_torch[0].cpu().numpy().astype(np.float32), result_torch[1].cpu().numpy().astype(np.float32))

        assert np.allclose(result_np[0], result_torch[0], atol=eps) and np.allclose(result_np[1], result_torch[1], atol=eps), "transform_sequence functions are not equivalent"

    def test_transform_batch():
        duration = 10
        fps = 30
        pad = 5
        batch_np = [np.random.rand(34, 9, 3) for _ in range(4)] 
        batch_torch = [torch.tensor(seq, dtype=torch.float32) for seq in batch_np]

        result_np = transform_batch(batch_np, duration, fps, pad)
        result_torch = transform_batch_torch(batch_torch, duration, fps, pad)
        result_torch = (result_torch[0].cpu().numpy().astype(np.float32), result_torch[1].cpu().numpy().astype(np.float32))
        assert np.allclose(result_np[0], result_torch[0], atol=eps) and np.allclose(result_np[1], result_torch[1], atol=eps), "transform_batch functions are not equivalent"

    # Run the tests
    test_resample_pose_seq()
    test_transform_sequence()
    test_transform_batch()