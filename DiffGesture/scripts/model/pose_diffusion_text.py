import torch
import torch.nn as nn

from .diffusion_net import *
from .diffusion_util import *

class PoseDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context

        # add attribute args for sampling
        self.args = args
        pose_dim = args.pose_dim
        diff_hidden_dim = args.diff_hidden_dim
        block_depth = args.block_depth

        self.in_size = 32 + pose_dim + 1 
        self.audio_encoder = WavEncoder()

        if args.text_encoder:
            self.text_encoder = TextEncoder()
        else:
            self.text_encoder = None

        try:
            self.unconditional = args.unconditional
        except:
            self.unconditional = False
        self.classifier_free = args.classifier_free
        assert (self.unconditional == self.classifier_free) or self.classifier_free, "Unconditional model implemented as classifier \
                                                                            free with probability of unconditional generation = 1"
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            if args.input_context == 'both':
                self.null_cond_emb_text = nn.Parameter(torch.randn(1,self.in_size))
                self.null_cond_emb_audio = nn.Parameter(torch.randn(1, self.in_size))
            else:
                self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))

        self.diffusion_net = DiffusionNet(
            net = TransformerModel( num_pose=args.n_poses,
                                    pose_dim=pose_dim, 
                                    embed_dim=pose_dim+3+self.in_size,
                                    hidden_dim=diff_hidden_dim,
                                    depth=block_depth//2,
                                    decoder_depth=block_depth//2
                                    ),
            var_sched = VarianceSchedule(
                num_steps=500,
                beta_1=1e-4,
                beta_T=0.02,
                mode='linear'
            )
        )

    def get_loss(self, x, pre_seq, in_audio, in_text = None):

        
        if self.input_context == 'both' :
            audio_feat_seq = self.audio_encoder(in_audio) # output (bs, n_frames, feat_size)
            text_feat_seq = self.text_encoder(in_text)
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)

        elif self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio) # output (bs, n_frames, feat_size)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        else:
            assert False


        if self.classifier_free:
            if args.input_context == 'both':
                mask_audio = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob_audio
                mask_text = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob_text
                mask_text = torch.where(mask_audio, 1, mask_text)*2 # If audio is masked, text is necessarly masked too
                mask = torch.cat([mask_text, mask_audio])
                in_data = torch.where(mask.unsqueeze(1).unsqueeze(2) == 1, self.null_cond_emb_audio.repeat(in_data.shape[1],1).unsqueeze(0), in_data)
                in_data = torch.where(mask.unsqueeze(1).unsqueeze(2) == 2, self.null_cond_emb_text.repeat(in_data.shape[1],1).unsqueeze(0), in_data)

            else:
                mask = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob
                in_data = torch.where(mask.unsqueeze(1).unsqueeze(2), self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0), in_data)
     

        neg_elbo = self.diffusion_net.get_loss(x, in_data)

        return neg_elbo
        
    def sample(self, pose_dim, pre_seq, in_audio, in_text=None):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)

        elif self.input_context == 'both':
            audio_feat_seq = self.audio_encoder(in_audio) 
            text_feat_seq = self.text_encoder(in_text)
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)


        if self.classifier_free:
            if self.input_context == 'both':
                uncondition_embedding_audio = self.null_cond_emb_audio.repeat(in_data.shape[1],1).unsqueeze(0)
                uncondition_embedding_text = self.null_cond_emb_text.repeat(in_data.shape[1],1).unsqueeze(0)
                samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, uncondition_embedding=uncondition_embedding, uncondition_embedding_text=uncondition_embedding_text, unconditional_sample=self.unconditional)
            else:
                uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
                samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, uncondition_embedding=uncondition_embedding, unconditional_sample=self.unconditional)
        else:
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim)
        return samples


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)


class TextEncoder(nn.Module):
    pass