from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib
import pprint
import sys
import time
import json

from DiffGesture.scripts.data_loader.lmdb_data_loader import *
from DiffGesture.scripts.model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from DiffGesture.scripts.model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from parse_args_trimodal import parse_args as parse_args
from DiffGesture.scripts.utils.average_meter import AverageMeter
from DiffGesture.scripts.utils.vocab_utils import build_vocab
import DiffGesture.scripts.utils.train_utils
from DiffGesture.scripts.model import vocab
import DiffGesture.scripts.model as model
from DiffGesture.scripts.train_eval.train_gan import train_iter_gan
import random
import torch.nn.functional as F
from DiffGesture_test_TED_2DVers import evaluate_testset

matplotlib.use('Agg')  # we don't use interactive GUI
[sys.path.append(i) for i in ['.', '..']]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    raise NotImplementedError("TO DO")

def init_model(args, lang_model=None, speaker_model=None, pose_dim=None, _device=None):
    # init model
    generator = discriminator = loss_fn = None
    if args.model == 'multimodal_context':
        n_frames = args.n_poses    
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)

        return generator, discriminator, loss_fn
    else:
        raise NotImplementedError("Change model configuration")


def train_epochs(args, train_data_loader, val_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    save_path = args.model_save_path + '/' + args.name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if args.restart_from_checkpoint:
        tb_writer = SummaryWriter(log_dir=args.tb_path)     
    else:  
        tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    with open(save_path + '/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 10



    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    generator, discriminator, loss_fn = init_model(args, lang_model, speaker_model, args.pose_dim, device)
    if args.restart_from_checkpoint:
        raise NotImplementedError("TO DO")
    else:
        last_epoch = 0
    
   
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None


    gen_optimizer = optim.Adam(generator.parameters(),lr=args.learning_rate, betas=(0.5, 0.999))
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))
    # training

    if args.dimension == 2:
        index = np.array([i for i in range(int(args.pose_dim * 1.5)) if i % 3 != 2])

    global_iter = last_epoch*len(train_data_loader)
    best_values = {}
    best_val_loss = (1e+10, 0)
    for epoch in range(last_epoch, args.epochs):

        
        val_metrics = evaluate_testset(test_data_loader=val_data_loader, generator=generator, embed_space_evaluator=embed_space_evaluator, args=args, pose_dim=args.pose_dim)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        val_loss = val_metrics['frechet']
        
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0 and epoch != last_epoch)  or epoch == args.epochs - 1:
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(save_path, args.name, epoch)

            DiffGesture.scripts.utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'state_dict': generator.state_dict(), 'disc_state_dict': discriminator.state_dict()
            }, save_name)

   

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)
            in_text = in_text.to(device) if in_text is not None else None
            in_spec = in_spec.to(device) if in_spec is not None else None
            in_text_padded = in_text_padded.to(device) if in_text_padded is not None else None
            
            
            batch_size = target_vec.size(0)
            if args.dimension == 2:
                # Extract the 2D version of the pose sequence 
                target_vec = target_vec[:, :, index]

            vid_indices = []
            if speaker_model and isinstance(speaker_model, vocab.Vocab):
                vids = aux_info['vid']
                vid_indices = [speaker_model.word2index[vid] for vid in vids]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            # train
            loss = []
           
            loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
                                    generator, discriminator,
                                    gen_optimizer, dis_optimizer)

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
    logging.info(f"Context {args.input_context}")

    collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec_3d).reshape(-1, 3)
    mean_pose = args.mean_pose_3d
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
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

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )
    selected = list(range(0,len(val_dataset),3))
    val_data = torch.utils.data.Subset(val_dataset, selected)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )                                      

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=mean_pose)

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, val_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    
    _args = parse_args()
    main({'args': _args})
