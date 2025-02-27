import torch
import copy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datasets.DGraphFin import DGraphFin
from torch_geometric.data import DataLoader
from models.roland.model import RolandGNN
from models.roland.lightning_modules import LightningGNN
from datetime import datetime

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

## config
hidden_conv1 = 16
hidden_conv2 = 16
lightning_root_dir = "experiments/roland/node_level"


def main():
    dataset = DGraphFin('data/DGraphFin',force_reload=True)
    for data_index in range(len(dataset) - 1):
        if data_index == 0:
            num_nodes = dataset.num_nodes
            previous_embeddings = [
                torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]),
                torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
        else:
            _, previous_embeddings = lightningModule.forward(train_data)
        snapshot = dataset[data_index]
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])

        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])

        train_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
        val_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
        train_indices = snapshot.node_mask.nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(train_indices))
        split_idx = int(0.9 * len(train_indices))
        train_mask[train_indices[perm[:split_idx]]] = True
        val_mask[train_indices[perm[split_idx:]]] = True
        train_data = snapshot.clone()
        train_data.node_mask = train_mask
        train_data.previous_embeddings = previous_embeddings

        val_data = snapshot.clone()
        val_data.node_mask = val_mask
        test_data = copy.deepcopy(dataset[data_index + 1])
        test_data.num_current_edges = test_data.num_edges
        test_data.num = test_data.num_nodes
        val_data.previous_embeddings = previous_embeddings
        test_data.previous_embeddings = previous_embeddings
        if snapshot.x is None:
            test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])

        model = RolandGNN(snapshot.x.shape[1], hidden_conv1, hidden_conv2, dataset.num_nodes)
        lightningModule = LightningGNN(model, learning_rate=0.01)
        experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        experiments_dir = f"{lightning_root_dir}/DGraphFin/{experiment_datetime}/index_{data_index}"
        csv_logger = CSVLogger(experiments_dir, version="")
        print(train_data)
        print(val_data)
        print(test_data)
        # Start training and testing.
        train_loader = DataLoader([train_data], batch_size=1)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)
        trainer = L.Trainer(default_root_dir=experiments_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")],
                            accelerator="auto",
                            devices="auto",
                            enable_progress_bar=True,
                            logger=csv_logger,
                            max_epochs=100
                            )
        trainer.fit(lightningModule, train_loader, val_loader)
        trainer.test(lightningModule, test_loader)


if __name__ == "__main__":
    main()