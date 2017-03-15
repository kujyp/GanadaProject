class Hyparms():
    def __init__(self):
        pass

HYPARMS = Hyparms()

HYPARMS.batch_size = 10
HYPARMS.learning_rate = 0.01
HYPARMS.max_steps = 3000
HYPARMS.log_dir = 'logs'
HYPARMS.train_data_dir = 'data/train'
HYPARMS.test_data_dir = 'data/test'
HYPARMS.recog_data_dir = 'data/recog'
HYPARMS.ckpt_dir = 'logs'
HYPARMS.ckpt_name = 'trained_weight'
HYPARMS.dropout_rate = 0.9