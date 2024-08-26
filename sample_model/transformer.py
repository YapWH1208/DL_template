import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, modelDim, maxLength=5000):
        super(PositionalEncoding, self).__init__()

        positionalEncoding = torch.zeros(maxLength, modelDim, device=DEVICE)

        position = torch.arange(0, maxLength, device=DEVICE, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, modelDim, 2, device=DEVICE, dtype=torch.float) * -(math.log(10000.0) / modelDim))

        positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
        positionalEncoding[:, 1::2] = torch.cos(position * divTerm)

        positionalEncoding = positionalEncoding.unsqueeze(0)
        self.register_buffer('positionalEncoding', positionalEncoding)

    def forward(self, x):
        x = x + self.positionalEncoding[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, modelDim, numHeads):
        super(MultiHeadAttention, self).__init__()
        
        assert modelDim % numHeads == 0, "modelDim must be divisible by numHeads, but modelDim: %d, numHeads: %d" % (modelDim, numHeads)
        
        self.modelDim = modelDim
        self.keyDim = modelDim // numHeads
        self.numHeads = numHeads

        self.W_q = nn.Linear(modelDim, modelDim)
        self.W_k = nn.Linear(modelDim, modelDim)
        self.W_v = nn.Linear(modelDim, modelDim)
        self.W_o = nn.Linear(modelDim, modelDim)
    
    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.keyDim)

        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention_probs, value)

        return out, attention_probs

    def split_heads(self, x):
        batchSize, seqLength, _ = x.size()
        return x.view(batchSize, seqLength, self.numHeads, self.keyDim).transpose(1, 2)

    def combine_heads(self, x:torch.Tensor):
        batchSize, _, seqLength, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, seqLength, self.modelDim)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
            mask = mask.unsqueeze(1)

        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        attention_out, attention_probs = self.attention(query, key, value, mask)

        out = self.combine_heads(attention_out)
        out = self.W_o(out)

        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, modelDim, hiddenDim):
        super(PositionWiseFeedForward, self).__init__()
        self.FF = nn.Sequential(
            nn.Linear(modelDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, modelDim)
        )
    
    def forward(self, x):
        out = self.FF(x)
        return out

################################################################################

class Encoder(nn.Module):
    def __init__(self, modelDim, numHeads, hiddenDim, dropout):
        super(Encoder, self).__init__()
        self.selfAttention = MultiHeadAttention(modelDim, numHeads)
        self.feedForward = PositionWiseFeedForward(modelDim, hiddenDim)
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.selfAttention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        feedForward_output = self.feedForward(x)
        x = self.norm2(x + self.dropout(feedForward_output))

        return x

class Decoder(nn.Module):
    def __init__(self, modelDim, numHeads, hiddenDim, dropout):
        super(Decoder, self).__init__()
        self.selfAttention = MultiHeadAttention(modelDim, numHeads)
        self.crossAttention = MultiHeadAttention(modelDim, numHeads)
        self.feedForward = PositionWiseFeedForward(modelDim, hiddenDim)
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)
        self.norm3 = nn.LayerNorm(modelDim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoderOutput, sourceMask, targetMask):
        attention_output = self.selfAttention(x, x, x, targetMask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.crossAttention(x, encoderOutput, encoderOutput, sourceMask)
        x = self.norm2(x + self.dropput(attention_output))
        feedForward_output = self.feedForward(x)
        x = self.norm3(x + self.dropout(feedForward_output))

        return x

################################################################################

class Transformer(nn.Module):
    def __init__(self, sourceVocabSize, targetVocabSize, modelDim, numHeads, numLayers, hiddenDim, maxLength, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(sourceVocabSize, modelDim)
        self.decoder_embedding = nn.Embedding(targetVocabSize, modelDim)
        self.positionalEmbedding = PositionalEncoding(modelDim, maxLength)

        self.encoder = nn.ModuleList([Encoder(modelDim, numHeads, hiddenDim, dropout) for _ in range(numLayers)])
        self.decoder = nn.ModuleList([Decoder(modelDim, numHeads, hiddenDim, dropout) for _ in range(numLayers)])

        self.fc = nn.Linear(modelDim, targetVocabSize)
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, source, target):
        sourceMask = (source != 0).unsqueeze(1).unsqueeze(2)
        targetMask = (target != 0).unsqueeze(1).unsqueeze(2)

        targetLength = target.size(1)
        no_peak_mask = (1- torch.triu(torch.ones(1, targetLength, targetLength), diagonal=1)).bool()
        targetMask = targetMask & no_peak_mask

        return sourceMask, targetMask

    def forward(self, source, target):
        sourceMask, targetMask = self.generate_mask(source, target)
        source_embedded = self.dropout(self.positionalEmbedding(self.encoder_embedding(source)))
        target_embedded = self.dropout(self.positionalEmbedding(self.decoder_embedding(target)))

        encoder_output = source_embedded
        for layer in self.encoder:
            encoder_output = layer(encoder_output, sourceMask)
        
        decoder_output = target_embedded
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output, sourceMask, targetMask)
        
        output = self.fc(decoder_output)

        return output