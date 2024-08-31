# This is a sample Python script.

import torch.nn as nn
import copy

class Transformer(nn.module):
    def __init__(self, encoder, decoder):
        super(Transformer,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, input_sentence):
        return self.encoder(input_sentence)

    def decode(self, context, current_sentence):
        return self.decoder(context, current_sentence)

    def forward(self, input_sentence, decoder_sentence):
        context = self.encode(input_sentence)
        output = self.decode(context,decoder_sentence)
        return output


class Encoder(nn.Module):
    def __init__(self,encoder_block,num_blocks):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block)] for _ in range(num_blocks))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,multihead_attention,position_feedforward):
        super(EncoderBlock,self).__init__()
        self.multihead_attention = multihead_attention
        self.position_feedforward = position_feedforward

    def forward(self, x):
        out = self.multihead_attention(x)
        out = self.position_feedforward(out)
        return out




def calculate_attention(query, key, value)












# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("main")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
