# This is a sample Python script.

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import copy


class Transformer(nn.module):
    def __init__(self, source_embed, target_embed, encoder, decoder, generator):
        super(Transformer,self).__init__()
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def encode(self, input_sentence, input_mask):
        return self.encoder(self.source_embed(input_sentence), input_mask)

    def decode(self, encoder_context, target_sentence, target_mask, input_target_mask):
        return self.decoder(encoder_context, self.target_embed(target_sentence), target_mask, input_target_mask)

    def forward(self, input_sentence, target_sentence):
        input_mask = self.make_src_mask(input_sentence)
        target_mask = self.make_tgt_mask(input_sentence)
        src_tgt_mask = self.make_src_tgt_mask(input_sentence, target_sentence)
        encoder_context = self.encode(input_sentence, input_mask)
        decoder_output = self.decode(encoder_context,target_sentence,target_mask,src_tgt_mask)
        out = self.generator(decoder_output)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_output

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask


class Encoder(nn.Module):
    def __init__(self, encoder_block,num_blocks):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block)] for _ in range(num_blocks))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,multihead_attention,position_feedforward):
        super(EncoderBlock,self).__init__()
        self.multihead_attention = multihead_attention
        self.position_feedforward = position_feedforward
        self.residuals = [ResidualConnectionNetwork() for _ in range(2)]

    def forward(self, x, mask):
        out = self.residuals[0](x, lambda x:self.multihead_attention(query=x,key=x,value=x,mask=mask))
        # out = self.multihead_attention(query=x, key=x, value=x, mask=mask)
        out = self.residuals[1](out, self.position_feedforward)
        return out





class MultiHeadAttentionLayer(nn.Module):
    # dim_model = dim_embed * h
    def __init__(self, dim_model, num_heads, qkv_fc_layer, out_fc_layer):
        super(MultiHeadAttentionLayer,self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.out_fc_layer = out_fc_layer

    def forward(self, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_k)
        num_batch = query.size(0)

        def transform_multihead(input_embedding, fc_layer):  # fc_layer : (dim_embed,dim_model)
            out = fc_layer(input_embedding)  # (n_batch, seq_len, d_embed)
            out = fc_layer.view(num_batch,-1,self.num_heads,self.dim_model//self.num_heads)  # (n_batch, seq_len, h, d_k)
            out = out.transpose(1,2)  # (n_batch, h, seq_len, d_k)
            return out

        query = transform_multihead(query,self.query_fc_layer)
        key = transform_multihead(key, self.key_fc_layer)
        value = transform_multihead(value, self.value_fc_layer)

        attention_value = self.calculate_attention(query,key,value,mask) # (n_batch, h, seq_len, d_k)
        attention_value = attention_value.transpose(1,2)  # (n_batch, seq_len, h, d_k)

        attention_value.view(num_batch, -1, self.dim_model)  # (n_batch, seq_len, dim_model)
        out = self.out_fc_layer(attention_value)  # (n_batch, seq_len, dim_embed)

        return out

    def make_pad_mask(self, query, key, pad_idx = 1):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    @staticmethod
    def calculate_attention(query, key, value, mask):
        # query, key, value: (n_batch, seq_len, d_k) // (n_batch, h, seq_len, d_k) when multihead
        # mask: (n_batch, seq_len, seq_len)

        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, seq_len, seq_len)
        out = torch.matmul(attention_prob, value)  # (n_batch, h, seq_len, d_k)
        return out


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc_layer_1, fc_layer_2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc_layer_1 = fc_layer_1 #(dim_embed, dim_ff)
        self.relu = nn.ReLU()
        self.fc_layer_2 = fc_layer_2 #(dim_ff, dim_embed)

    def forward(self, x):
        out = self.fc_layer_1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ResidualConnectionNetwork(nn.Module):
    def __init__(self):
        super(ResidualConnectionNetwork, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out

# Teacher forcing for decoder
def make_subsequent_mask(query, key):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    return mask

def make_target_mask(self, target):
    pad_mask = self.make_pad_mask(target, target)
    seq_mask = self.make_subsequent_mask(target, target)
    mask = pad_mask & seq_mask
    return pad_mask & seq_mask


class Decoder(nn.Module):
    def __init__(self, decoder_block, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.num_layers)])

    def forward(self, encoder_context, target_sentence, target_mask, input_target_mask):
        for layer in self.layers:
            target_sentence = layer(target_sentence,encoder_context,target_mask,input_target_mask)
        return target_sentence

#copy
class DecoderBlock(nn.Module):
    def __init__(self, multihead_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.multihead_attention = multihead_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionNetwork() for _ in range(3)]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.multihead_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)

    def forward(self, x):
        out = self.embedding(x)
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


def build_model(src_vocab_size, tgt_vocab_size, device=torch.device("cpu"), max_len=256, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                   d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)

    src_embed = TransformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed))
    tgt_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed))

    attention = MultiHeadAttentionLayer(
                                        d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed))
    position_ff = PositionWiseFeedForwardLayer(
                                               fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(
                                 self_attention = copy(attention),
                                 position_ff = copy(position_ff))
    decoder_block = DecoderBlock(
                                 self_attention = copy(attention),
                                 cross_attention = copy(attention),
                                 position_ff = copy(position_ff))

    encoder = Encoder(
                      encoder_block = encoder_block,
                      n_layer = n_layer)
    decoder = Decoder(
                      decoder_block = decoder_block,
                      n_layer = n_layer)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("main")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
