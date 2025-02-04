# Configuration
class Config:
    def __init__(self):
        self.num_workers = 8
        self.input_size = 1024
        self.hidden_size = 128
        self.num_layers = 2
        self.num_heads = 4
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.bidirectional = False
        self.dropout = 0.3
        self.weight_decay = 1e-4
        self.lr_patience = 5
        self.patience = 20
        self.root_data_dir = '../../data/my_data/protbert_embedding_with_label/trim'


config = Config()
