# Character-Based Named Entity Recognition in Keras
## Using a Bi-Directional LSTM Recurrent Neural Network

#### Keras implementation based on models by:

 * Kuru, Onur, Ozan Arkan Can, and Deniz Yuret. [*CharNER: Character-Level Named Entity Recognition.*](http://www.aclweb.org/anthology/C/C16/C16-1087.pdf)
 
 * Klein, D., Smarr, J., Nguyen, H., & Manning, C. D. (2003, May). [*Named entity recognition with character-level models. In Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4 (pp. 180-183). Association for Computational Linguistics.*](http://nlp.stanford.edu/manning/papers/conll-ner.pdf)
  
## Usage

- Implement `get_texts()`, `get_labels()` and `get_x_y()` or `get_x_y_generator()` with your own data source. 

    - `x` is a tensor of shape: `(batch_size, max_length)`.
        Entries in dimension 1 are alphabet indices, index 0 is the padding symbol.
        
    - `y` is a tensor of shape: `(batch_size, max_length, number_of_labels)`.
        Entries in dimension 2 are label indices, index 0 is the null label.

- Tweak the model hyper-parameters in `config.py` 
     
- Run `train.py`