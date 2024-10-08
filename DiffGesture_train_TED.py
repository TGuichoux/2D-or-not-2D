from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib
import pprint
import sys
import time

from DiffGesture.scripts.data_loader.lmdb_data_loader import *
from DiffGesture.scripts.model.pose_diffusion import PoseDiffusion
from parse_args_diffusion import parse_args
from DiffGesture.scripts.train_eval.train_diffusion import train_iter_diffusion
from DiffGesture.scripts.utils.average_meter import AverageMeter
from DiffGesture.scripts.utils.vocab_utils import build_vocab
import DiffGesture.scripts.utils.train_utils

matplotlib.use('Agg')  # we don't use interactive GUI
[sys.path.append(i) for i in ['.', '..']]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    print("init diffusion model")
    diffusion = PoseDiffusion(args).to(_device)

    diffusion.load_state_dict(checkpoint['state_dict'])

    return args, diffusion, lang_model, speaker_model, pose_dim, epoch

def init_model(args, _device):
    # init model
    if args.model == 'pose_diffusion':
        print("init diffusion model")
        diffusion = PoseDiffusion(args).to(_device)
    return diffusion


def train_epochs(args, train_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.restart_from_checkpoint:
        tb_writer = SummaryWriter(log_dir=args.tb_path)     
    else:  
        tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 50

    diffusion_model = init_model(args, device)
    if args.restart_from_checkpoint:
        args, diffusion_model, _, _, _, last_epoch = load_checkpoint_and_model(args.checkpoint_path, device)
    else:
        last_epoch = 0

    optimizer = optim.Adam(diffusion_model.parameters(),lr=args.learning_rate, betas=(0.5, 0.999))

    # training

    if args.dimension == 2:
        index = np.array([i for i in range(int(args.pose_dim * 1.5)) if i % 3 != 2])

    global_iter = last_epoch*len(train_data_loader)
    for epoch in range(last_epoch, args.epochs):

        # save model
        if (epoch % save_model_epoch_interval == 0 and epoch > 0 and epoch != last_epoch) or epoch == args.epochs - 1:

            state_dict = diffusion_model.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            DiffGesture.scripts.utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'state_dict': state_dict,
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            _, _, _, target_pose, target_vec, in_audio, _, _ = data
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)
            batch_size = target_vec.size(0)
            if args.dimension == 2:
                # Extract the 2D version of the pose sequence
                target_vec = target_vec[:, :, index]
 
            

            # train
            loss = []
            if args.model == 'pose_diffusion':
                loss = train_iter_diffusion(args, in_audio, target_vec, 
                                      diffusion_model, optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, DiffGesture.scripts.utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)
            iter_start_time = time.time()

    tb_writer.close()


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        DiffGesture.scripts.utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    DiffGesture.scripts.utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec_3d).reshape(-1, 3)
    mean_pose = args.mean_pose_3d
    train_dataset = SpeechMotionDataset(args.root_data_dir[0]+args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=mean_pose,
                                        remove_word_timing=(args.input_context == 'text')
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.root_data_dir[0]+args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )

    test_dataset = SpeechMotionDataset(args.root_data_dir[0]+args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=mean_pose)

    # build vocab
    vocab_cache_path = os.path.join(args.root_data_dir[0], 'TED/vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.root_data_dir[0]+args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
