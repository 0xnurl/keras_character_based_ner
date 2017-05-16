class Config:
    sentence_max_length = 200
    input_dropout = 0.3
    output_dropout = 0.5
    recurrent_stack_depth = 5
    batch_size = 32
    max_epochs = 100
    learning_rate = 0.001
    embed_size = 256
    num_lstm_units = 128
    early_stopping = 2