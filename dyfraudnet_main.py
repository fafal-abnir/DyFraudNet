import argparse
import torch
import copy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from datasets.DGraphFin import DGraphFin
from torch_geometric.data import DataLoader
from models.dyfraudnet.model import DyFraudNet
from models.dyfraudnet.lightning_modules import LightningGNN
from datetime import datetime
from utils.visualization import visualize_embeddings

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser(description="DyFraudNetGNN Training Arguments")
    parser.add_argument("--enable_memory", action="store_true", help="Enable the memory for GNN")
    parser.add_argument("--fresh_start", action="store_true", help="retraining from scratch on each timestamp")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate(default:0.01")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("--memory_size", type=int, default=128,
                        help="Size of memory for evolving weights (default: 128)")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GAT", "GCN"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--graph_window_size", type=int, default=1, help="the size of graph window size")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows for running the experiment")
    return parser.parse_args()


lightning_root_dir = "experiments/dyfraudnet/node_level"


def main():
    args = get_args()
    # Model arguments
    enable_memory = args.enable_memory
    fresh_start = args.fresh_start
    hidden_size = args.hidden_size
    memory_size = args.memory_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    gnn_type = args.gnn_type
    # Data arguments
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    model = None
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    dataset = DGraphFin('data/DGraphFin', force_reload=True, edge_window_size=graph_window_size,
                        num_windows=num_windows)
    for data_index in range(len(dataset) - 1):
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

        val_data = snapshot.clone()
        val_data.node_mask = val_mask
        test_data = copy.deepcopy(dataset[data_index + 1])
        test_data.num_current_edges = test_data.num_edges
        test_data.num = test_data.num_nodes
        if snapshot.x is None:
            test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
        if (model is None) or fresh_start:
            model = DyFraudNet(snapshot.x.shape[1], memory_size=memory_size, hidden_size=hidden_size, out_put_size=2,
                               gnn_type=gnn_type, enable_memory=enable_memory)
        lightningModule = LightningGNN(model, learning_rate=learning_rate)
        experiments_dir = f"{lightning_root_dir}/DGraphFin/{graph_window_size}_days/M{enable_memory}_{gnn_type}_F{fresh_start}/{experiment_datetime}/index_{data_index} "
        csv_logger = CSVLogger(experiments_dir, version="")
        csv_logger.log_hyperparams(vars(args))
        print(f"Time Index: {data_index}")
        print(train_data)
        print(val_data)
        print(test_data)
        # Start training and testing.
        train_loader = DataLoader([train_data], batch_size=1)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)
        # Callbacks
        model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")
        trainer = L.Trainer(default_root_dir=experiments_dir,
                            callbacks=[model_checkpoint],
                            accelerator="auto",
                            devices="auto",
                            enable_progress_bar=True,
                            logger=csv_logger,
                            max_epochs=epochs
                            )
        # Visualization embedding before training
        print("start visualization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        train_data = train_data.to(device)
        train_labels = train_data.y[train_data.node_mask]
        train_labels_np = train_labels.cpu().numpy()
        with torch.no_grad():
            all_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
            train_embeddings_np = all_embeddings[train_data.node_mask].cpu().numpy()
        train_label_colors = ['blue' if label == 0 else 'red' for label in train_labels_np]
        visualize_embeddings(train_embeddings_np, train_label_colors, 0,
                             f'{experiments_dir}/train_embeddings_not_trained.png')

        trainer.fit(lightningModule, train_loader, val_loader)
        trainer.test(lightningModule, test_loader)

        # Visualization embedding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        test_labels = test_data.y[test_data.node_mask]
        test_labels_np = test_labels.cpu().numpy()
        with torch.no_grad():
            train_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
            train_embeddings_np = train_embeddings[train_data.node_mask].cpu().numpy()
            test_embeddings = model.get_embedding(test_data.x, test_data.edge_index)
            test_embeddings_np = test_embeddings[test_data.node_mask].cpu().numpy()
        # train_label_colors = ['blue' if label == 0 else 'red' for label in train_data.y.cpu().numpy()]
        test_label_colors = ['blue' if label == 0 else 'red' for label in test_labels_np]
        visualize_embeddings(train_embeddings_np, train_label_colors, epochs,
                             f'{experiments_dir}/train_embedding_trained.png')
        visualize_embeddings(test_embeddings_np, test_label_colors, 'None', f'{experiments_dir}/test_embedding.png')


if __name__ == "__main__":
    main()
