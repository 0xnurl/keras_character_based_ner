import numpy as np
from alphabet import CharBasedNERAlphabet


class CharBasedNERDataset:
    NULL_LABEL = '0'
    BASE_LABELS = [NULL_LABEL]

    def __init__(self):
        self.texts = self.get_texts()
        self.alphabet = CharBasedNERAlphabet(self.texts)
        self.labels = self.BASE_LABELS + self.get_lables()
        self.num_labels = len(self.labels)
        self.num_to_label = {}
        self.label_to_num = {}

        self.init_mappings()

    def get_texts(self):
        """ Implement with own data source. """
        raise NotImplementedError

    def get_x_y(self, sentence_maxlen, dataset_name='all'):
        """ Implement with own data source.

        :param sentence_maxlen: maximum number of characters per sample
        :param dataset_name: 'all', 'train', 'dev' or 'test'
        :return: Tuple (x, y)
                x: Array of shape (batch_size, sentence_maxlen). Entries in dimension 1 are alphabet indices, index 0 is the padding symbol
                y: Array of shape (batch_size, sentence_maxlen, self.num_labels). Entries in dimension 2 are label indices, index 0 is the null label
        """
        raise NotImplementedError

    def get_x_y_generator(self, sentence_maxlen, dataset_name='all'):
        """ Implement with own data source.

        :return: Generator object that yields tuples (x, y), same as in get_x_y()
        """
        raise NotImplementedError

    def get_labels(self):
        """ Implement with own data source.

        :return: List of labels (classes) to predict, e.g. 'PER', 'LOC', not including the null label '0'.
        """
        raise NotImplementedError

    def str_to_x(self, s, maxlen):
        x = np.zeros(maxlen)
        for c, char in enumerate(s[:maxlen]):
            x[c] = self.alphabet.get_char_index(char)
        return x.reshape((-1, maxlen))

    def x_to_str(self, x):
        return [[self.alphabet.num_to_char[i] for i in row] for row in x]

    def y_to_labels(self, y):
        Y = []
        for row in y:
            Y.append([self.num_to_label[np.argmax(one_hot_labels)] for one_hot_labels in row])
        return Y

    def init_mappings(self):
        for num, label in enumerate(self.labels):
            self.num_to_label[num] = label
            self.label_to_num[label] = num
