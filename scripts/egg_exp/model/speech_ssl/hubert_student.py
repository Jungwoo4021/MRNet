from typing import Optional, Tuple, Union
 
import math
import numpy as np
import torch
import torch.nn as nn
import transformers
import transformers.models.hubert.modeling_hubert as hubert
from transformers import AutoConfig, HubertConfig, HubertModel
from transformers.modeling_outputs import BaseModelOutput

__all__ = ['StudentHubert_DistHubt', 'StudentHubert_FitHubt']

BASE_960 = 'facebook/hubert-base-ls960'



############
## Models ##
############
class StudentHubert_DistHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size, return_all_hiddens=False, init_teacher_param=None, ft_adapter_hidden_size=None):
        super(StudentHubert_DistHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = AutoConfig.from_pretrained(BASE_960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size
        self.hubert = CustomHubertModel(config, ft_adapter_hidden_size=ft_adapter_hidden_size)
        
        # weight initialization
        teacher = HubertModel.from_pretrained(
            BASE_960,
            from_tf=bool(".ckpt" in BASE_960),
            config=AutoConfig.from_pretrained(BASE_960),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.hubert.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict(), strict=False)
        self.hubert.feature_projection.load_state_dict(teacher.feature_projection.state_dict(), strict=False)
        if init_teacher_param is not None:
            for i in range(num_hidden_layer):
                self.hubert.encoder.layers[i].load_state_dict(teacher.encoder.layers[init_teacher_param[i]].state_dict(), strict=False)
        
    def forward(self, x, idx_without_adapter=None):
        x = self.hubert(x, output_hidden_states=self.return_all_hiddens, idx_without_adapter=idx_without_adapter)
        
        if self.return_all_hiddens:
            return torch.stack(x.hidden_states, dim=1)
        else:
            return x.last_hidden_state

class StudentHubert_FitHubt(nn.Module):
    def __init__(self, num_hidden_layer, hidden_size_s, hidden_size_t, sequence_length, return_all_hiddens=False, ft_adapter_hidden_size=None):
        super(StudentHubert_FitHubt, self).__init__()
        self.return_all_hiddens = return_all_hiddens

        # set transformer encoder
        config = HubertConfig.from_pretrained(BASE_960)
        config.num_hidden_layers = num_hidden_layer
        config.hidden_size = hidden_size_s
        config.num_feat_extract_layers = 9
        config.conv_dim = (128, 256, 256, 256, 256, 256, 512, 512, 512)
        config.conv_stride = (5, 1, 2, 2, 2, 2, 1, 2, 2)
        config.conv_kernel =  (10, 1, 3, 3, 3, 3, 1, 2, 2)
        config.intermediate_size = hidden_size_s
        config.do_stable_layer_norm = True
        self.hubert = CustomHubertModel(config, use_time_reduction=True, ft_adapter_hidden_size=ft_adapter_hidden_size)

        self.hidden_size_s = hidden_size_s
        self.hidden_size_t = hidden_size_t
        self.sequence_length = sequence_length
        self.sequence_length_st = int(sequence_length / 2)

        for i in range(0, 12):
            setattr(self, 'conv_{}'.format(i+1), nn.Conv1d(self.sequence_length_st, self.sequence_length, kernel_size=1, stride=1, bias=False))
            setattr(self, 'fc_{}'.format(i+1), nn.Linear(self.hidden_size_s, self.hidden_size_t))

    def forward(self, x, idx_without_adapter=None):
        x = self.hubert(x, output_hidden_states=self.return_all_hiddens, idx_without_adapter=idx_without_adapter)
        
        if self.return_all_hiddens:
            x = torch.stack(x.hidden_states, dim=1)
            bs, l, _, _ = x.size()

            x_h = []

            # transform student output
            for i in range(1, l):
                x_s = getattr(self, 'conv_{}'.format(i))(x[:, i, :, :])
                x_s = x_s.reshape(bs * self.sequence_length, -1)
                x_s = getattr(self, 'fc_{}'.format(i))(x_s)
                
                x_h.append(x_s.reshape(bs, self.sequence_length, -1))

            return torch.stack(x_h, dim=1)
        else:
            raise NotImplementedError()


#################
## Sub modules ##
#################
class CustomHubertModel(hubert.HubertModel):
    def __init__(self, config, use_time_reduction=False, ft_adapter_hidden_size=None):
        super().__init__(config)

        self.config = config
        self.feature_extractor = hubert.HubertFeatureEncoder(config)
        self.feature_projection = hubert.HubertFeatureProjection(config)
        self.use_time_reduction = use_time_reduction
        if use_time_reduction:
            self.time_reduction_layer = nn.Conv1d(config.conv_dim[-1], config.conv_dim[-1], kernel_size=2, stride=2, bias=False)
        
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        
        if config.do_stable_layer_norm:
            self.encoder = CustomHubertEncoderStableLayerNorm(config, ft_adapter_hidden_size)
        else:
            self.encoder = CustomHubertEncoder(config, ft_adapter_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        idx_without_adapter: Optional[list] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CNN
        extract_features = self.feature_extractor(input_values)
        
        # time reduction layer (for fithubert)
        if self.use_time_reduction:
            extract_features = self.time_reduction_layer(extract_features)
        
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idx_without_adapter=idx_without_adapter,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
class CustomHubertEncoder(nn.Module):
    def __init__(self, config, ft_adapter_hidden_size=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = hubert.HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([CustomHubertEncoderLayer(config, ft_adapter_hidden_size=ft_adapter_hidden_size) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        idx_without_adapter=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, idx_without_adapter=idx_without_adapter
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CustomHubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config, ft_adapter_hidden_size=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = hubert.HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [CustomHubertEncoderLayerStableLayerNorm(config, ft_adapter_hidden_size=ft_adapter_hidden_size) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        idx_without_adapter=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, idx_without_adapter=idx_without_adapter
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
class CustomHubertEncoderLayer(nn.Module):
    def __init__(self, config, ft_adapter_hidden_size=None):
        super().__init__()
        self.attention = hubert.HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = hubert.HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # FT adapter
        self.use_ft_adapter = ft_adapter_hidden_size is not None
        if self.use_ft_adapter:
            self.ft_adapter = FTAdapter(config.hidden_size, ft_adapter_hidden_size)
            
    def forward(self, hidden_states, attention_mask=None, output_attentions=False, idx_without_adapter=None):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # FT adapter
        if self.use_ft_adapter:
            if idx_without_adapter is None:
                h = self.ft_adapter(hidden_states)    
            else:
                h_non_adapter, h_adapter = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                h_non_adapter = h_non_adapter * 0
                h_adapter = self.ft_adapter(h_adapter)
                h = torch.cat((h_non_adapter, h_adapter), dim=0)
            hidden_states = hidden_states + self.feed_forward(hidden_states) + h
        else:
            hidden_states = hidden_states + self.feed_forward(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CustomHubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config, ft_adapter_hidden_size=None):
        super().__init__()
        self.attention = hubert.HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = hubert.HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # FT adapter
        self.use_ft_adapter = ft_adapter_hidden_size is not None
        if self.use_ft_adapter:
            self.ft_adapter = FTAdapter(config.hidden_size, ft_adapter_hidden_size)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        idx_without_adapter = None
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        # FT adapter
        if self.use_ft_adapter:
            if idx_without_adapter is None:
                h = self.ft_adapter(hidden_states)    
            else:
                h_non_adapter, h_adapter = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                h_non_adapter = h_non_adapter * 0
                h_adapter = self.ft_adapter(h_adapter)
                h = torch.cat((h_non_adapter, h_adapter), dim=0)
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states)) + h
        else:
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class FTAdapter(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.down = nn.Linear(in_channels, hidden_channels)
        self.non_linear_func = nn.ReLU()
        self.up = nn.Linear(hidden_channels, in_channels)
        self.scale = nn.Parameter(torch.ones(1))
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linear_func(x)
        x = self.up(x)
        
        x = x * self.scale
        
        return x