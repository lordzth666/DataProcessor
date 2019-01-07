class DataSet:
    def __init__(self,
                 source_path=None,
                 save_base_path=None,
                 filename="train_batch_",
                 num_batches=1):
        self.source_path = source_path
        self.save_base_path = save_base_path
        self.filename = filename
        self.num_batches = num_batches
        pass


    def process(self):
        raise NotImplementedError

