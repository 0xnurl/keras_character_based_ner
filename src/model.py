import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


class CharacterBasedLSTMModel:
    """ Character-based stacked bi-directional LSTM model
    Based on: `Kuru, Onur, Ozan Arkan Can, and Deniz Yuret. "CharNER: Character-Level Named Entity Recognition.`
    """

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.model = self.get_model()

    def get_model(self):
        num_words = len(self.dataset.alphabet)
        num_labels = len(self.dataset.labels)

        model = Sequential()

        model.add(Embedding(num_words,
                            self.config.embed_size,
                            mask_zero=True))
        model.add(Dropout(self.config.input_dropout))

        for _ in range(self.config.recurrent_stack_depth):
            model.add(Bidirectional(LSTM(self.config.num_lstm_units, return_sequences=True)))

        model.add(Dropout(self.config.output_dropout))
        model.add(TimeDistributed(Dense(num_labels, activation='softmax')))

        # TODO Add Viterbi decoder here, see Kuru et al.

        optimizer = Adam(lr=self.config.learning_rate,
                         clipnorm=1.0)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', self.non_null_label_accuracy])
        return model

    def fit(self):
        x_train, y_train = self.dataset.get_x_y(self.config.sentence_max_length, dataset_name='train')
        x_dev, y_dev = self.dataset.get_x_y(self.config.sentence_max_length, dataset_name='dev')

        early_stopping = EarlyStopping(patience=self.config.early_stopping,
                                       verbose=1)
        checkpointer = ModelCheckpoint(filepath="/tmp/model.weights.hdf5",
                                       verbose=1,
                                       save_best_only=True)

        self.model.fit(x_train,
                       y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.max_epochs,
                       validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=[early_stopping, checkpointer])

    def fit_generator(self):
        train_data_generator = self.dataset.get_x_y_generator(dataset_name='train',
                                                              maxlen=self.config.sentence_max_length,
                                                              batch_size=self.config.batch_size)
        dev_data_generator = self.dataset.get_x_y_generator(dataset_name='dev',
                                                            maxlen=self.config.sentence_max_length,
                                                            batch_size=self.config.batch_size)
        early_stopping = EarlyStopping(patience=self.config.early_stopping,
                                       verbose=1)

        self.model.fit_generator(train_data_generator,
                                 steps_per_epoch=self.dataset.num_train_docs / self.config.batch_size,
                                 epochs=self.config.max_epochs,
                                 validation_data=dev_data_generator,
                                 validation_steps=self.dataset.num_dev_docs / self.config.batch_size,
                                 callbacks=[early_stopping]
                                 )

    def evaluate(self):
        x_test, y_test = self.dataset.get_x_y(self.config.sentence_max_length, dataset_name='test')
        self.model.evaluate(x_test, y_test, batch_size=self.config.batch_size)

    def evaluate_generator(self):
        test_data_generator = self.dataset.get_x_y_generator(dataset_name='test',
                                                             maxlen=self.config.sentence_max_length,
                                                             batch_size=self.config.batch_size)

        self.model.evaluate_generator(test_data_generator, steps=self.dataset.num_test_docs / self.config.batch_size)

    def predict_str(self, s):
        """ Get model prediction for a string
        :param s: string to get named entities for
        :return: a list of len(s) tuples: [(character, predicted-label for character), ...]
        """
        x = self.dataset.str_to_x(s, self.config.sentence_max_length)
        predicted_classes = self.predict_x(x)
        chars = self.dataset.x_to_str(x)[0]
        labels = self.dataset.y_to_labels(predicted_classes)[0]

        return list(zip(chars, labels))

    def predict_x(self, x):
        return self.model.predict(x, batch_size=1)

    @staticmethod
    def non_null_label_accuracy(y_true, y_pred):
        """Calculate accuracy excluding null-label targets (index 0).
        Useful when the null label is over-represented in the data, like in Named Entity Recognition tasks.

        typical y shape: (batch_size, sentence_length, num_labels)
        """

        y_true_argmax = K.argmax(y_true, -1)  # ==> (batch_size, sentence_length, 1)
        y_pred_argmax = K.argmax(y_pred, -1)  # ==> (batch_size, sentence_length, 1)

        y_true_argmax_flat = tf.reshape(y_true_argmax, [-1])
        y_pred_argmax_flat = tf.reshape(y_pred_argmax, [-1])

        non_null_targets_bool = K.not_equal(y_true_argmax_flat, K.zeros_like(y_true_argmax_flat))
        non_null_target_idx = K.flatten(K.cast(tf.where(non_null_targets_bool), 'int32'))

        y_true_without_null = K.gather(y_true_argmax_flat, non_null_target_idx)
        y_pred_without_null = K.gather(y_pred_argmax_flat, non_null_target_idx)

        mean = K.mean(K.cast(K.equal(y_pred_without_null,
                                     y_true_without_null),
                             K.floatx()))

        # If the model contains a masked layer, Keras forces metric output to have same shape as y:
        fake_shape_mean = K.ones_like(y_true_argmax, K.floatx()) * mean
        return fake_shape_mean

    def get_custom_objects(self):
        return {'non_null_label_accuracy': self.non_null_label_accuracy}
