import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TimeLoggerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.train_start_time
        trainer.logger.log_metrics({"train_time_sec": elapsed_time})
        print(f"Training Time: {elapsed_time:.2f} seconds")

    def on_test_start(self, trainer, pl_module):
        self.test_start_time = time.time()

    def on_test_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.test_start_time
        trainer.logger.log_metrics({"test_time_sec": elapsed_time})
        print(f"Testing Time: {elapsed_time:.2f} seconds")
