import pickle
import sys

class Config:
    def __init__(self, argv):
        self.type = argv[1]
        self.source_path = argv[2]
        self.save_base_path = argv[3]
        self.filename = "train_batch_"
        self.num_batches = 5
