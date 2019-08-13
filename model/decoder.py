import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from component import *

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, corpus_size, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(corpus_size, input_size)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bias = True,  batch_first=True, bidirectional = False)
        self.linear = nn.Linear(hidden_size, corpus_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        # this is for custom calcualting caption length
        # we only have vector data here, so we need to bring it from somewhere else
        # lst = list(map(lambda t : len(t), captions))
        # custom_length = torch.tensor(lst)

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # self.lstm.flatten_parameters() # this is for data parrallel
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def update_layer(self, corpus_size):
        self.embed = nn.Embedding(corpus_size, self.input_size)
        self.linear = nn.Linear(self.hidden_size, corpus_size)

    def sample(self, features, states = None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, corpus_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, input_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, input_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def beam(self, features, label_encoder, k):

        beam_search = BeamSearch(k)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):

            if i == 0:
                hiddens, states = self.lstm(inputs, None)       # hiddens: (batch_size, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1))       # outputs:  (batch_size, corpus_size)
                probability, index = outputs.max(1)             # predicted: (batch_size)

                aa = index.cpu().numpy()
                inverted_label = label_encoder.inverse_transform(aa)

                features = self.embed(index)  # inputs: (batch_size, input_size)
                features = features.unsqueeze(1)  # inputs: (batch_size, 1, input_size)

                beam_search.create_start_node(probability[0], index[0], features, states, label_encoder)
                continue

            for phrase in beam_search.phrases:

                hiddens, states = self.lstm(phrase.get_features(), phrase.get_state())
                outputs = self.linear(hiddens.squeeze(1))
                probability_list, index_list = torch.topk(outputs, beam_search.k)

                for probability, index in zip(probability_list[0], index_list[0]):

                    features = self.embed(index.unsqueeze(0))
                    features = features.unsqueeze(1)

                    beam_search.new_phrases.append(phrase.nodes + [BeamNode(probability, index, features, states, label_encoder)])

                if i == 1:
                    break

            beam_search.run_selection()

            if beam_search.check_end():
                break

        return beam_search.phrases[0].get_hypothesis()
