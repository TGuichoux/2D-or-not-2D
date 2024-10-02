from transformers import Data2VecAudioConfig, Data2VecAudioModel, AutoProcessor
from scipy.signal import resample_poly

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]
        
        

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # speech
                 speech_context_length: int,
                 speech_size:int, 
                 speech_transformer_width: int,
                 speech_transformer_heads: int,
                 speech_transformer_layers: int,
                 # motion
                 motion_size: int, 
                 motion_context_length: int,
                 vocab_size: int,
                 motion_transformer_width: int,
                 motion_transformer_heads: int,
                 motion_transformer_layers: int
                 ):
        super().__init__()

        #config = Data2VecAudioConfig()
        self.speech_pre_encoder = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h").to("cuda")
        self.speech_pre_encoder.eval()
        #self.speech_config = self.speech_pre_encoder.config

        self.speech_context_length = speech_context_length
        self.motion_context_length = motion_context_length
     
        self.speech_transformer = Transformer(
            width=speech_transformer_width,
            layers=speech_transformer_layers,
            heads=speech_transformer_heads,
            attn_mask=self.build_attention_mask_speech()
        )

        self.motion_transformer = Transformer(
            width=motion_transformer_width,
            layers=motion_transformer_layers,
            heads=motion_transformer_heads,
            attn_mask=self.build_attention_mask_motion()
        )

        self.in_proj_motion = nn.Linear(motion_size, motion_transformer_width)
        self.in_proj_speech = nn.Linear(speech_size, speech_transformer_width)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, motion_transformer_width) if vocab_size is not None else None
        self.motion_positional_embedding = PositionalEmbedding(motion_transformer_width)
        self.speech_positional_embedding = PositionalEmbedding(speech_transformer_width)

        self.motion_ln_final = LayerNorm(motion_transformer_width)
        self.speech_ln_final = LayerNorm(speech_transformer_width)

        self.motion_projection = nn.Parameter(torch.empty(motion_transformer_width, embed_dim))
        self.speech_projection = nn.Parameter(torch.empty(speech_transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        
        #nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.speech_positional_embedding, std=0.01)
        # nn.init.normal_(self.motion_positional_embedding, std=0.01)


        proj_std = (self.motion_transformer.width ** -0.5) * ((2 * self.motion_transformer.layers) ** -0.5)
        attn_std = self.motion_transformer.width ** -0.5
        fc_std = (2 * self.motion_transformer.width) ** -0.5
        for block in self.motion_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.motion_projection is not None:
            nn.init.normal_(self.motion_projection, std=self.motion_transformer.width ** -0.5)

        proj_std = (self.speech_transformer.width ** -0.5) * ((2 * self.speech_transformer.layers) ** -0.5)
        attn_std = self.speech_transformer.width ** -0.5
        fc_std = (2 * self.speech_transformer.width) ** -0.5
        for block in self.speech_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.speech_projection is not None:
            nn.init.normal_(self.speech_projection, std=self.speech_transformer.width ** -0.5)


    def build_attention_mask_speech(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.speech_context_length, self.speech_context_length)
        # mask.fill_(float("-inf"))
        # mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def build_attention_mask_motion(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.motion_context_length, self.motion_context_length)
        # mask.fill_(float("-inf"))
        # mask.triu_(1)  # zero out the lower diagonal
        return mask
        

    @property
    def dtype(self):
        return torch.float32

    def encode_speech(self, speech):
        speech = {"input_values":speech}
        speech = self.speech_pre_encoder(**speech).last_hidden_state

        speech = self.in_proj_speech(speech)
        pos_seq = torch.arange(speech.shape[1], device=speech.device).to(speech.dtype)
        pos_emb = self.speech_positional_embedding(pos_seq) 
        
        x = speech + pos_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.speech_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #x = self.speech_ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
       
        #x = x[torch.arange(x.shape[0]), speech.argmax(dim=-1).argmax(dim=-1)] 
        x = x @ self.speech_projection
        return x

    def encode_motion(self, motion):
        #x = self.token_embedding(motion).type(self.dtype)  # [batch_size, n_ctx, d_model]

        motion = self.in_proj_motion(motion.type(self.dtype))
        pos_seq = torch.arange(motion.shape[1], device=motion.device).to(motion.dtype)
        pos_emb = self.motion_positional_embedding(pos_seq) 
        
        x = motion + pos_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.motion_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #x = self.motion_ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #x = x[torch.arange(x.shape[0]), motion.argmax(dim=-1).argmax(dim=-1)] 
        x = x @ self.motion_projection

        return x

    def forward(self, speech, motion):

        motion_feature = self.encode_motion(motion).mean(dim=1)
        speech_features = self.encode_speech(speech).mean(dim=1)

        # normalized features
        motion_feature = motion_feature / motion_feature.norm(dim=1, keepdim=True)
        speech_features = speech_features / speech_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_feature @ speech_features.t()
        logits_per_speech = logits_per_motion.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_motion, logits_per_speech


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["motion_projection", "proj", "speech_projection"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    
    embed_dim = state_dict["motion_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        transformer_width, transformer_heads, transformer_layers,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


if __name__ == "__main__":
    config = Data2VecAudioConfig()
    model = Data2VecAudioModel(config)

    motion_batch = torch.rand((64,34,27)).to("cuda")
    speech_batch = torch.rand((12,36267)).to("cuda")

    clip_model = CLIP(
        embed_dim=32,
        speech_context_length=113,
        speech_transformer_width=768,
        speech_size=768,
        speech_transformer_heads=8,
        speech_transformer_layers=12,
        motion_context_length=34,
        vocab_size=None,
        motion_size=27,
        motion_transformer_width=768,
        motion_transformer_heads=8,
        motion_transformer_layers=12
    ).to("cuda")

    out = clip_model.encode_speech(speech_batch)
    out_motion = clip_model.encode_motion(motion_batch)
    print(out.shape, out_motion.shape)

    speech_feature, motion_features = clip_model(speech_batch, motion_batch)
    print(speech_feature.shape, motion_features.shape)
    assert False
    processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h").to("cuda")

    inputs = {"input_values":speech_batch}
    
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    res_out = resample_poly(last_hidden_states.detach().cpu().numpy(),up=3, down=5, axis=1)
    print(res_out.shape)
