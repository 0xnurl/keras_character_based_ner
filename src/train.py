from config import Config
from dataset import CharBasedNERDataset
from model import CharacterBasedLSTMModel

if __name__ == '__main__':
    config = Config()
    dataset = CharBasedNERDataset()
    model = CharacterBasedLSTMModel(config, dataset)

    model.fit()
    model.evaluate()
    print(model.predict_str('La nostalgie n’a rien d’un sentiment esthétique, elle n’est même pas liée non plus au souvenir d’un bonheur, on est nostalgique d’un endroit simplement parce qu’on y a vécu, bien ou mal peu importe, le passé est toujours beau, et le futur aussi d’ailleurs, il n’y a que le présent qui fasse mal, qu’on transporte avec soi comme un abcès de souffrance qui vous accompagne entre deux infinis de bonheur paisible'))
