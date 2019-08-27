import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from component import *
from config import *

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, corpus_size, num_layers, max_seq_length=30, use_bi_direct = False, device = None):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bi_direct = use_bi_direct
        self.max_seg_length = max_seq_length
        self.device = device

        self.embed = nn.Embedding(corpus_size, input_size)
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bias = True,
                            batch_first = True,
                            dropout = rnn_lstm_dropout,
                            bidirectional = self.use_bi_direct)
        if self.use_bi_direct:
            print("LSTM bidireciton:", self.use_bi_direct)
            self.linear = nn.Linear(hidden_size*2, corpus_size)   # 2 for bidirection
        else:
            self.linear = nn.Linear(hidden_size, corpus_size)

    def forward(self, features, x, l):

        # for i in range(len(x)):
        new_dataset = []
        before_embd = []

        for idx, item in enumerate(x):
            before_embd.append(item["data"])

        section_tensor = torch.stack(before_embd, 0)
        embeddings = self.embed(section_tensor)
        embeddings_list = list(embeddings.cpu().detach().numpy())

        for row, item in enumerate(x):

            image_tensor = features[item["section"]]
            embeddings_row = embeddings_list[row]

            image_tensor2 = list(image_tensor.cpu().detach().numpy())

            tmp = list(embeddings_row)
            tmp.insert(0, image_tensor2)
            new_dataset.append(tmp.copy())

        new_dataset_tensor = torch.FloatTensor(new_dataset)
        packed = pack_padded_sequence(new_dataset_tensor, l, batch_first=True)

        hiddens, _ = self.lstm(packed)

        sent_output = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)[0]

        aaaaaaa = sent_output[:, -1, :]  # <- this is 윤재
        result = self.linear(aaaaaaa)

        return result

    # def forward(self, features, attributes, captions, lengths):
    #     embeddings = self.embed(captions)
    #     if cnn_output_combine_methods == 3:                      # inserting VCAP at each state of LSTM sequencing
    #         embeddings = embeddings * (attributes.unsqueeze(1))
    #     embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    #     packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    #
    #     # if self.use_bi_direct:
    #     #      # Set initial states
    #     #     h0 = torch.zeros(self.num_layers*2, embeddings.size(0), self.hidden_size).to(self.device) # 2 for bidirection
    #     #     c0 = torch.zeros(self.num_layers*2, embeddings.size(0), self.hidden_size).to(self.device)
    #     #
    #     #     # Forward propagate LSTM
    #     #     # out, _ = self.lstm(packed, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
    #     #     hiddens, _ = self.lstm(packed, (h0, c0))
    #     #
    #     #     # lets try adding a dropout layer later on
    #     #     # out = self.dropout(out)
    #     #
    #     #     # Decode the hidden state of the last time step
    #     #     # out = self.fc(out[:, -1, :])
    #     #     # return out
    #     #     outputs = self.linear(hiddens[:, -1, :])
    #     #     # outputs = self.linear(hiddens[0])
    #     #     return outputs
    #
    #     hiddens, _ = self.lstm(packed)
    #     outputs = self.linear(hiddens[0])
    #     return outputs

    def update_layer(self, corpus_size):
        self.embed = nn.Embedding(corpus_size, self.input_size)
        self.linear = nn.Linear(self.hidden_size, corpus_size)

    def sample(self, features, attributes, states = None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if cnn_output_combine_methods == 3:                      # inserting VCAP at each state of LSTM sequencing
                inputs = inputs * (attributes.unsqueeze(1))
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def beam(self, features, attributes, label_encoder, k):
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
                inputs = inputs * (attributes.unsqueeze(1))

                beam_search.create_start_node(probability[0], index[0], features, states, label_encoder)
                continue

            for phrase in beam_search.phrases:
                hiddens, states = self.lstm(phrase.get_features(), phrase.get_state())
                outputs = self.linear(hiddens.squeeze(1))
                probability_list, index_list = torch.topk(outputs, beam_search.k)

                for probability, index in zip(probability_list[0], index_list[0]):

                    features = self.embed(index.unsqueeze(0))
                    features = features.unsqueeze(1)
                    inputs = inputs * (attributes.unsqueeze(1))

                    beam_search.new_phrases.append(phrase.nodes + [BeamNode(probability, index, features, states, label_encoder)])

                if i == 1:
                    break

            beam_search.run_selection()

            if beam_search.check_end():
                break
        return beam_search.phrases[0].get_hypothesis()
