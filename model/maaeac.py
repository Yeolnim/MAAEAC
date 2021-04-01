# -*- coding: utf-8 -*-
# file: maaeacc.py
# author: Yue Jian
# Copyright (C) 2021. All Rights Reserved.


from pytorch_transformers.modeling_bert import BertForTokenClassification, BertPooler
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import copy
import numpy as np
import math


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, config, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.args.max_seq_length))
        zero_tensor = torch.tensor(zero_vec).float().to(self.args.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class MAAEAC(BertForTokenClassification):

    def __init__(self, bert_base_model, args):
        super(MAAEAC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert_for_global_context = bert_base_model
        self.args = args
        # do not init lcf layer if BERT-SPC or BERT-BASE specified
        # if self.args.local_context_focus in {'cdw', 'cdm', 'fusion'}:
        if not self.args.use_unique_bert:
            self.bert_for_local_context = copy.deepcopy(self.bert_for_global_context)
        else:
            self.bert_for_local_context = self.bert_for_global_context
        self.pooler = BertPooler(config)
        if args.dataset in {'camera', 'car', 'phone', 'notebook'}:
            self.dense = torch.nn.Linear(768, 2)
        else:
            self.dense = torch.nn.Linear(768, 3)
        self.bert_global_focus = self.bert_for_global_context
        self.dropout = nn.Dropout(self.args.dropout)
        self.SA1 = SelfAttention(config, args)
        self.SA2 = SelfAttention(config, args)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)
        self.hidden = config.hidden_size

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.args.device)

    def get_batch_polarities(self, b_polarities):
        b_polarities = b_polarities.detach().cpu().numpy()
        shape = b_polarities.shape
        polarities = np.zeros((shape[0]))
        i = 0
        for polarity in b_polarities:
            polarity_idx = np.flatnonzero(polarity + 1)
            try:
                polarities[i] = polarity[polarity_idx[0]]
            except:
                pass
            i += 1
        polarities = torch.from_numpy(polarities).long().to(self.args.device)
        return polarities

    def feature_dynamic_mask(self, text_local_indices, aspect_indices, distances_input=None):
        texts = text_local_indices.cpu().numpy()  # batch_size x seq_len
        asps = aspect_indices.cpu().numpy()  # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        # mask_len = self.args.SDRD
        mask_len = self.args.SDRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.args.max_seq_length, self.hidden), dtype=np.float32)  # batch_size x seq_len x hidden size
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):  # For each sample
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i])  # Calculate aspect length
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin):  # Masking to the left
                    masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, self.args.max_seq_length):  # Masking to the right
                    masked_text_raw_indices[text_i][j] = np.zeros((self.hidden), dtype=np.float)
            else:
                distances_i = distances_input[text_i]
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.args.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices, distances_input=None):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.args.max_seq_length, self.hidden), dtype=np.float32)  # batch x seq x dim
        mask_len = self.args.SDRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2  # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                    sdrd = abs(i - asp_avg_index) + asp_len / 2
                    if sdrd > self.args.SDRD:
                        distances[i] = 1 - (sdrd - self.args.SDRD) / np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i]  # distances of batch i-th
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances_i[i] = 1
                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.args.device)

    def get_ids_for_local_context_extractor(self, text_indices):
        # convert BERT-SPC input to BERT-BASE format
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.args.device)

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None, valid_ids=None, attention_mask_label=None,  distance_to_aspect=None, aspect_indices=None):
        global cat_out, mean_pool
        if not self.args.use_bert_spc:
            input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
            labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out, _ = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)
        polarity_labels = self.get_batch_polarities(polarities)

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)

        if self.args.local_context_focus is not None:

            if self.args.use_bert_spc:
                local_context_ids = self.get_ids_for_local_context_extractor(input_ids_spc)
            else:
                local_context_ids = input_ids_spc

            local_context_out, _ = self.bert_for_local_context(input_ids_spc)
            batch_size, max_len, feat_dim = local_context_out.shape
            local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        local_valid_output[i][jj] = local_context_out[i][j]
            local_context_out = self.dropout(local_valid_output)

            if 'cdm' in self.args.local_context_focus:
                cdm_vec = self.feature_dynamic_mask(local_context_ids, aspect_indices, distance_to_aspect)  
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdm_context_out = self.SA1(cdm_context_out)
                cat_out = torch.cat((global_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif 'cdw' in self.args.local_context_focus:
                cdw_vec = self.feature_dynamic_weighted(local_context_ids, aspect_indices, distance_to_aspect)  
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cdw_context_out = self.SA1(cdw_context_out)
                cat_out = torch.cat((global_context_out, cdw_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif 'fusion' in self.args.local_context_focus:
                cdm_vec = self.feature_dynamic_mask(local_context_ids, aspect_indices, distance_to_aspect)  
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdw_vec = self.feature_dynamic_weighted(local_context_ids, aspect_indices, distance_to_aspect)  
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cat_out = torch.cat((global_context_out, cdw_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_triple(cat_out)
            sa_out = self.SA2(cat_out)
            pooled_out = self.pooler(sa_out)
        else:
            pooled_out = self.pooler(global_context_out)
        pooled_out = self.dropout(pooled_out)
        apc_logits = self.dense(pooled_out)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_sen = CrossEntropyLoss()
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))
            loss_apc = loss_sen(apc_logits, polarity_labels)

            return loss_ate, loss_apc
        else:
            return ate_logits, apc_logits
