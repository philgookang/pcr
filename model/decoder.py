import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

# class Decoder(nn.Module):
#
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
#         """Set the hyper-parameters and build the layers."""
#         super(Decoder, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.max_seg_length = max_seq_length
#
#     def forward(self, features, captions, lengths):
#         """Decode image feature vectors and generates captions."""
#
#         # this is for custom calcualting caption length
#         # we only have vector data here, so we need to bring it from somewhere else
#         # lst = list(map(lambda t : len(t), captions))
#         # custom_length = torch.tensor(lst)
#
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
#         # self.lstm.flatten_parameters() # this is for data parrallel
#         hiddens, _ = self.lstm(packed)
#         outputs = self.linear(hiddens[0])
#
#         return outputs
#
#     def update_layer(self, vocab_size):
#         self.embed = nn.Embedding(vocab_size, self.embed_size)
#         self.linear = nn.Linear(self.hidden_size, vocab_size)
#
#     def sample(self, features, states=None):
#         """Generate captions for given image features using greedy search."""
#         sampled_ids = []
#         inputs = features.unsqueeze(1)
#         for i in range(self.max_seg_length):
#             hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
#             outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
#             _, predicted = outputs.max(1)  # predicted: (batch_size)
#             sampled_ids.append(predicted)
#             inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
#             inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
#         sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
#         return sampled_ids


class StatefulLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.out_size = hidden_size
        self.hidden = None
        self.cell = None

    def reset_state(self):
        self.hidden = None
        self.cell = None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.hidden is None:
            state_size = [batch_size, self.out_size]
            self.cell = Variable(torch.zeros(state_size)).cuda()
            self.hidden = Variable(torch.zeros(state_size)).cuda()
        self.hidden, self.cell = self.lstm(x, (self.hidden, self.cell))

        return self.hidden


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train == False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = self.m / (1 - dropout)

        return mask * x


class LSTMBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(LSTMBlock, self).__init__()
        self.lstm = StatefulLSTM(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.locked_dropout = LockedDropout()

    def reset_state(self):
        self.lstm.reset_state()
        self.locked_dropout.reset_state()

    def forward(self, x, train=True):
        x = self.lstm(x)
        x = self.bn(x)
        x = self.locked_dropout(x, train=train)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20, stateful=True):
        super(Decoder, self).__init__()
        self.stateful = stateful
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        if stateful:
            self.lstm_block1 = LSTMBlock(embed_size, hidden_size)
            if self.num_layers >= 2:
                self.lstm_blocks = nn.ModuleList([LSTMBlock(hidden_size, hidden_size) for i in range(num_layers-1)])
        # self.lstm1 = StatefulLSTM(embed_size, hidden_size)
        # self.bn_lstm1 = nn.BatchNorm1d(hidden_size)
        # self.dropout1 = LockedDropout()
        # self.lstm2 = StatefulLSTM(hidden_size, hidden_size)
        # self.bn_lstm2 = nn.BatchNorm1d(hidden_size)
        # self.dropout2 = LockedDropout()
        # self.lstm3 = StatefulLSTM(hidden_size, hidden_size)
        # self.bn_lstm3 = nn.BatchNorm1d(hidden_size)
        # self.dropout3 = LockedDropout()
        else:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def reset_state(self):
        if self.stateful:
            self.lstm_block1.reset_state()
            if self.num_layers >= 2:
                for i, _ in enumerate(self.lstm_blocks):
                    self.lstm_blocks[i].reset_state()


    def forward(self, features, captions, lengths, train=True):
        """Decode image feature vectors and generates captions."""
        self.train = train
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        if self.stateful:
            no_of_timesteps = embeddings.shape[1]
            self.reset_state()
            outputs = []
            for i in range(no_of_timesteps):
                h = self.lstm_block1(embeddings[:, i, :], train = train)
                #print('h shape', h.shape)
                if self.num_layers >= 2:
                    for i, _ in enumerate(self.lstm_blocks):
                        h = self.lstm_blocks[i](h, train = train)
                #print('h shape2', h.shape)
                outputs.append(h)

            outputs = torch.stack(outputs)  # (time_steps,batch_size,features)
            # outputs = outputs.permute(1, 2, 0)  # (batch_size,features,time_steps) [3, 256, 17]

            # Jazzik Nov. 18, 12:16pm
            outputs = outputs.permute(1, 0, 2) # (batch_size, time_steps, features)
            packed_outputs = []
            for i in range(outputs.shape[0]):
                packed_outputs.append(outputs[i][:lengths[i]])
            packed_outputs = torch.cat(packed_outputs, 0)

            return self.linear(packed_outputs)
            #h = self.dropout(h)

        else:
            # Test on batch_size = 3, embeddings.shape = [3, 17, 256], where
            # 17 the max(lengths) + 1
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed)
            # return hiddens
            # hiddens here is a PackedSequence, hiddens[0] is the data tensor
            # hiddens[1] is the batch_size, tensor
            outputs = self.linear(hiddens[0])
            return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
