from functools import reduce
import torch

class BeamSearch:

    def __init__(self, k):
        self.k = k
        self.phrases = []
        self.new_phrases = []

    def create_start_node(self, probability, index, features, states, label_encoder):
        for i in range(self.k):
            self.phrases.append(BeamPhrase(probability, index, features, states, label_encoder))

    def run_selection(self):

        tmp = []

        for i,node_list in enumerate(self.new_phrases):
            probability = 1
            for node in node_list: probability *= node.probability
            tmp.append((i, probability))

        tmp_sorted = sorted(tmp, key=lambda tup: tup[1], reverse = True)

        for i,phrase in enumerate(self.phrases):
            phrase.nodes = self.new_phrases[tmp_sorted[i][0]]
            # phrase.print_to_screen()

        self.new_phrases.clear()

    def check_end(self):
        for phrase in self.phrases:
            if not phrase.has_end():
                return False
        return True


class BeamPhrase:

    def __init__(self, probability, index, features, states, label_encoder):
        self.nodes = [BeamNode(probability, index, features, states, label_encoder)]

    def get_features(self):
        node = self.nodes[len(self.nodes)-1]
        return node.features

    def get_state(self):
        node = self.nodes[len(self.nodes)-1]
        return node.states

    def get_word(self):
        node = self.nodes[len(self.nodes)-1]
        return node.word

    def has_end(self):
        for node in self.nodes:
            if node.word == "<end>":
                return True
        return False

    def get_hypothesis(self):
        return_list = []
        for node in self.nodes:
            if node.word == "<start>":
                continue
            elif node.word == "<end>" or node.word == ".":
                break
            return_list.append(node.word)
        return return_list

    def print_to_screen(self):
        s = ""
        for node in self.nodes:
            s += node.word + " "
        print(s)


class BeamNode:

    def __init__(self, probability, index, features, states, label_encoder):
        self.probability = torch.tensor([probability])
        self.index = torch.tensor([index])
        self.features = features
        self.states = states
        self.word = label_encoder.inverse_transform(self.index)[0]
