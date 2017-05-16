class CharBasedNERAlphabet:
    PADDING_SYMBOL = '<PAD>'
    UNKNOWN_CHAR_SYMBOL = '<UNK>'
    BASE_ALPHABET = [PADDING_SYMBOL, UNKNOWN_CHAR_SYMBOL]

    def __init__(self, texts):
        self.characters = self.BASE_ALPHABET + self.get_alphabet_from_texts(texts)
        self.char_to_num = None
        self.num_to_char = None
        self.init_mappings()

    def get_alphabet_from_texts(self, texts):
        all_characters = set()

        for t in texts:
            text_characters = set(t)
            all_characters |= text_characters

        alphabet = sorted(list(all_characters))
        return alphabet

    def init_mappings(self):
        self.char_to_num = self.get_char_to_num()
        self.num_to_char = self.get_num_to_char()

    def get_char_to_num(self):
        return {char: c for c, char in enumerate(self.characters)}

    def get_num_to_char(self):
        return {c: char for c, char in enumerate(self.characters)}

    def get_char_index(self, char):
        try:
            num = self.char_to_num[char]
        except KeyError:
            num = self.char_to_num[self.UNKNOWN_CHAR_SYMBOL]
        return num

    def __str__(self):
        return str(self.characters)

    def __len__(self):
        return len(self.characters)

    def __iter__(self):
        return self.characters.__iter__()
