import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.base import ResidualBase, PositionalEncoding


class FEARec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout)

        self.n_layers = max(2, self.depth)
        self.layer = nn.ModuleList()
        for i in range(self.n_layers):
            self.layer_ramp = FEABlock(self.n_heads, self.embedding_k, i, self.dropout, self.n_layers, self.max_seq_len)
            self.layer.append(self.layer_ramp)

    def encode_user(self, hist_item_idx, user_idx=None):
        seqs = self.item_embedding(hist_item_idx)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)
        for layer_module in self.layer:
            hidden_states = layer_module(seqs)
        return hidden_states[:,-1,:]


class FEABlock(nn.Module):
    def __init__(
        self, n_heads, hidden_size, i, dropout, n_layers, max_seq_len
    ):
        super(FEABlock, self).__init__()
        self.hybrid_attention = HybridAttention(
            n_heads, hidden_size, i, dropout, n_layers, max_seq_len
        )
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, hidden_states):
        attention_output = self.hybrid_attention(hidden_states)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class HybridAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, i, dropout, n_layers, max_seq_len):
        super(HybridAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.factor = 5
        self.scale = None
        self.mask_flag = True
        self.output_attention = False
        self.filter_mixer = None
        self.global_ratio = 0.6
        self.spatial_ratio = 0.1
        self.n_layers = n_layers
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(dropout)
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.out_dropout = nn.Dropout(dropout)

        if self.global_ratio > (1 / self.n_layers):
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'G'
        else:
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'L'
        self.max_item_list_length = max_seq_len
        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.global_ratio)) // (self.n_layers - 1)
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)

        if self.filter_mixer == 'G':
            self.w = self.global_ratio
            self.s = self.slide_step

        if self.filter_mixer == 'L':
            self.w = self.local_ratio
            self.s = self.filter_size

        self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (i * self.s))
        self.right = int((self.max_item_list_length // 2 + 1) - i * self.s)

        self.q_index = list(range(self.left, self.right))
        self.k_index = list(range(self.left, self.right))
        self.v_index = list(range(self.left, self.right))
        # if sample in time domain
        self.std = True
        if self.std:
            self.time_q_index = self.q_index
            self.time_k_index = self.k_index
            self.time_v_index = self.v_index
        else:
            self.time_q_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_k_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_v_index = list(range(self.max_item_list_length // 2 + 1))

        print('modes_q={}, index_q={}'.format(len(self.q_index), self.q_index))
        print('modes_k={}, index_k={}'.format(len(self.k_index), self.k_index))
        print('modes_v={}, index_v={}'.format(len(self.v_index), self.v_index))


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [batch_size, sequence_length, num_heads, embedding_k]
        return x

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, input_tensor):
        mixed_query_layer = self.query_layer(input_tensor)
        mixed_key_layer = self.key_layer(input_tensor)
        mixed_value_layer = self.value_layer(input_tensor)

        queries = self.transpose_for_scores(mixed_query_layer)
        keys = self.transpose_for_scores(mixed_key_layer)
        values = self.transpose_for_scores(mixed_value_layer)
        
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        q_fft_box = torch.zeros(B, H, E, len(self.q_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            q_fft_box[:, :, :, i] = q_fft[:, :, :, j]

        k_fft_box = torch.zeros(B, H, E, len(self.k_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            k_fft_box[:, :, :, i] = k_fft[:, :, :, j]

        res = q_fft_box * torch.conj(k_fft_box)
        box_res = torch.zeros(B, H, E, L // 2 + 1,  device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            box_res[:, :, :, j] = res[:, :, :, i]

        corr = torch.fft.irfft(box_res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        context_layer = V.view(*new_context_layer_shape)

        # q
        q_fft_box = torch.zeros(B, H, E, len(self.time_q_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_q_index):
            q_fft_box[:, :, :, i] = q_fft[:, :, :, j]
        spatial_q = torch.zeros(B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_q_index):
            spatial_q[:, :, :, j] = q_fft_box[:, :, :, i]

        # k
        k_fft_box = torch.zeros(B, H, E, len(self.time_k_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_k_index):
            k_fft_box[:, :, :, i] = k_fft[:, :, :, j]
        spatial_k = torch.zeros(B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_k_index):
            spatial_k[:, :, :, j] = k_fft_box[:, :, :, i]

        # v
        v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
        v_fft_box = torch.zeros(B, H, E, len(self.time_v_index), device=v_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_v_index):
            v_fft_box[:, :, :, i] = v_fft[:, :, :, j]
        spatial_v = torch.zeros(B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.time_v_index):
            spatial_v[:, :, :, j] = v_fft_box[:, :, :, i]

        queries = torch.fft.irfft(spatial_q, dim=-1)
        keys = torch.fft.irfft(spatial_k, dim=-1)
        values = torch.fft.irfft(spatial_v, dim=-1)

        queries = queries.permute(0, 1, 3, 2)
        keys = keys.permute(0, 1, 3, 2)
        values = values.permute(0, 1, 3, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        qkv = torch.matmul(attention_probs, values)  # [256, 2, index, 32]
        context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer_spatial.size()[:-2] + (self.all_head_size,)
        context_layer_spatial = context_layer_spatial.view(*new_context_layer_shape)
        context_layer = (1 - self.spatial_ratio) * context_layer + self.spatial_ratio * context_layer_spatial

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(dropout_rate)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.gelu(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
