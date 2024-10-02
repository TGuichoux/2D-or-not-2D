import numpy as np
import torch
import math
import random
import time
from DiffGesture.scripts.data_loader.data_preprocessor import DataPreprocessor


def generate_gestures(args, generator, lang_model, audio, words, pose_dim, audio_sr=16000,
                      seed_seq=None, fade_out=False, vid=None, device='cpu'):
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    # pre seq
    #pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec_2d) + 1))
    pre_seq = torch.zeros((1, n_frames, args.pose_dim+1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    if args.model == 'multimodal_context':
        if args.z_type == 'speaker':
            if not vid:
                vid = random.randrange(pose_decoder.z_obj.n_words)
            print('vid:', vid)
            vid = torch.LongTensor([vid]).to(device)
        else:
            vid = None

    print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    

    out_dir_vec = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)
        print(' ')

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)

        if args.model == 'pose_diffusion':
            out_dir_vec = generator.sample(pose_dim, pre_seq, in_audio)

        if args.model ==  'multimodal_context':
            out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid)



        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_dir_vec = np.vstack(out_list)

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        #out_dir_vec[end_frame-n_smooth:] = np.zeros((len(args.mean_dir_vec_2d)))  # fade out to mean poses
        out_dir_vec[end_frame-n_smooth:] = np.zeros((args.pose_dim))

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y


    return out_dir_vec






def generate_gestures_trimodal(args, pose_decoder, lang_model, audio, words, audio_sr=16000, vid=None,
                      seed_seq=None, fade_out=False):
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    use_spectrogram = False
    if args.model == 'speech2gesture':
        use_spectrogram = True

    # pre seq
    pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

    sr = 16000
    spectrogram = None
    if use_spectrogram:
        # audio to spectrogram
        spectrogram = extract_melspectrogram(audio, sr)

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    spectrogram_sample_length = int(round(unit_time * sr / 512))
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    # prepare speaker input
    if args.z_type == 'speaker':
        if not vid:
            vid = random.randrange(pose_decoder.z_obj.n_words)
        print('vid:', vid)
        vid = torch.LongTensor([vid]).to(device)
    else:
        vid = None

    print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    out_dir_vec = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare spectrogram input
        in_spec = None
        if use_spectrogram:
            # prepare spec input
            audio_start = math.floor(start_time / clip_length * spectrogram.shape[0])
            audio_end = audio_start + spectrogram_sample_length
            in_spec = spectrogram[:, audio_start:audio_end]
            in_spec = torch.from_numpy(in_spec).unsqueeze(0).to(device)

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        print(' ')
        in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)
        pre_seq_partial = pre_seq[0, 0:args.n_pre_poses, :-1].unsqueeze(0)

        # synthesize
        print(in_text_padded)
        if args.model == 'multimodal_context':
            out_dir_vec, *_ = pose_decoder(pre_seq, in_text_padded, in_audio, vid)
        elif args.model == 'joint_embedding':
            _, _, _, _, _, _, out_dir_vec = pose_decoder(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
        elif args.model == 'seq2seq':
            words_lengths = torch.LongTensor([in_text.shape[1]]).to(device)
            out_dir_vec = pose_decoder(in_text, words_lengths, pre_seq_partial, None)
        elif args.model == 'speech2gesture':
            out_dir_vec = pose_decoder(in_spec, pre_seq_partial)
        else:
            assert False

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_dir_vec = np.vstack(out_list)

    # additional interpolation for seq2seq
    if args.model == 'seq2seq':
        n_smooth = args.n_pre_poses
        for i in range(num_subdivision):
            start_frame = args.n_pre_poses + i * (args.n_poses - args.n_pre_poses) - n_smooth
            if start_frame < 0:
                start_frame = 0
                end_frame = start_frame + n_smooth * 2
            else:
                end_frame = start_frame + n_smooth * 3

            # spline interp
            y = out_dir_vec[start_frame:end_frame]
            x = np.array(range(0, y.shape[0]))
            w = np.ones(len(y))
            w[0] = 5
            w[-1] = 5

            coeffs = np.polyfit(x, y, 3)
            fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
            interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
            interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

            out_dir_vec[start_frame:end_frame] = interpolated_y

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        out_dir_vec[end_frame-n_smooth:] = np.zeros((len(args.mean_dir_vec)))  # fade out to mean poses

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec