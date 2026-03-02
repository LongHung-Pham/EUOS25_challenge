import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    BatchNorm1d,
    ReLU,
    Dropout,
    LayerNorm,
    ModuleList,
)

from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn import GlobalAttention


def GINConvLayer(layer_count, node_dim=52, edge_dim=14, h_dim=256):
    if layer_count == 0:
        h = Sequential(Linear(node_dim, h_dim), ReLU(), Linear(h_dim, h_dim))
    else:
        h = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, h_dim))
    conv = GINEConv(h, edge_dim=edge_dim)
    return conv


class GNN_net(torch.nn.Module):
    def __init__(
        self,
        num_gnn_layers=5,
        graph_pooling="sum",
        JK="concat",
        node_dim=52,
        edge_dim=14,
        h_dim=512,
        ffn_dim=128,
        drop_ratio=0.2,
        task_heads=None,
    ):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of GIN layers
            task_heads: Dictionary specifying task names and their output dimensions (default: {'y_sol': 1, 'y_logd': 1, 'y_clint': 1})
        """
        super(GNN_net, self).__init__()

        if task_heads is None:
            self.task_heads = {"y_sol": 1, "y_logd": 1, "y_clint": 1}
        else:
            self.task_heads = task_heads

        self.num_gnn_layers = num_gnn_layers
        self.JK = JK
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_gnn_layers):
            self.gnns.append(GINConvLayer(layer, node_dim, edge_dim, h_dim))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_gnn_layers):
            self.batch_norms.append(BatchNorm1d(h_dim))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=Linear((self.num_gnn_layers) * h_dim + self.node_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=Linear(h_dim, 1))

        self.ffn_dim = ffn_dim
        self.h_dim = h_dim
        self.drop_ratio = drop_ratio

        # Shared FFN layers
        if self.JK == "concat":
            self.shared_ffn = Sequential(
                Linear(self.node_dim + (self.num_gnn_layers) * h_dim, self.ffn_dim * 4),
                ReLU(),
                Dropout(0.3),  # original 0.3
                Linear(self.ffn_dim * 4, self.ffn_dim),
                ReLU(),
                Dropout(0.2),  # original 0.2
            )
        else:
            self.shared_ffn = Sequential(
                Linear(h_dim, self.ffn_dim * 2),
                ReLU(),
                Dropout(0.3),
                Linear(self.ffn_dim * 2, self.ffn_dim),
                ReLU(),
                Dropout(0.2),
            )
        # Task-specific prediction heads
        self.task_predictors = nn.ModuleDict()
        for task_name, output_dim in self.task_heads.items():
            self._create_task_head(task_name, output_dim)
        # Current active task for fine-tuning (None means all tasks are active)
        self.active_task = None

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        h_list = [x]
        for layer in range(self.num_gnn_layers):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # Relu and Dropout at each layer, but not relu at the last layer
            if layer == self.num_gnn_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Jumping knowledge
        if self.JK == "concat":
            node_out = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_out = h_list[-1]
        elif self.JK == "max":
            h_list = [
                h.unsqueeze_(0) for h in h_list[1:]
            ]  # Add an extra dimension at position 0
            node_out = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list[1:]]
            node_out = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        # Pooling
        x = self.pool(node_out, batch)  # graph output
        x = self.shared_ffn(x)
        # Task-specific predictions
        outputs = {}
        # If in fine-tuning mode, use only the active task
        if self.active_task is not None:
            outputs[self.active_task] = self.task_predictors[self.active_task](x)
            return outputs[
                self.active_task
            ]  # Return just the prediction for the active task
        # Otherwise return predictions for all tasks
        for task_name in self.task_heads:
            outputs[task_name] = self.task_predictors[task_name](x)

        return outputs

    def _create_task_head(self, task_name, output_dim=1):
        """Create a task-specific prediction head.

        Args:
            task_name: Name of the task
            output_dim: Output dimension (default: 1 for regression)
        """
        self.task_predictors[task_name] = nn.Sequential(
            nn.Linear(self.ffn_dim, self.ffn_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.ffn_dim // 4, output_dim),
        )
        # Update task_heads dictionary
        self.task_heads[task_name] = output_dim

    def add_task(self, task_name, output_dim=1):
        """Add a new task head to the model.
        Args:
            task_name: Name of the new task
            output_dim: Output dimension for the new task
        """
        if task_name in self.task_heads:
            print(f"Task {task_name} already exists. Skipping.")
            return
        self._create_task_head(task_name, output_dim)
        print(f"Added new task head: {task_name}")

    def set_fine_tuning_mode(self, task_name=None):
        """Set the model to fine-tuning mode for a specific task.

        Args:
            task_name: The name of the task to fine-tune on.
                       If None, all tasks are active (pre-training mode)
        """
        if task_name is not None and task_name not in self.task_heads:
            raise ValueError(
                f"Task '{task_name}' not found. Available tasks: {list(self.task_heads.keys())}. "
                f"Use add_task('{task_name}') to add a new task before fine-tuning."
            )
        self.active_task = task_name

        if task_name is not None:
            # Freeze shared parameters during fine-tuning
            for param in self.gnns.parameters():
                param.requires_grad = True
            for param in self.batch_norms.parameters():
                param.requires_grad = True

            for param in self.shared_ffn.parameters():
                param.requires_grad = True  # also train the shared FFN layers

            # Unfreeze only the relevant task head
            for name, param in self.task_predictors.named_parameters():
                if task_name in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # In pre-training mode, all parameters are trainable
            for param in self.parameters():
                param.requires_grad = True


class GPS_net(torch.nn.Module):
    """Graph GPS Network for multitask learning.

    Uses GPSConv layers (local GINEConv MPNN + global multi-head attention)
    as the backbone instead of plain GINEConv.

    Dimension notes
    ---------------
    Raw node features (node_dim=52) and edge features (edge_dim=14) are
    projected to h_dim via learned Linear + LayerNorm before the GPS stack.
    All GPS layers then operate in h_dim → h_dim, keeping channel dims
    consistent (a requirement of GPSConv).  LayerNorm is used on the input
    projections following Graph-Transformer best practices (batch-size
    independent, consistent between train/inference).

    JK (Jumping Knowledge) aggregates only the GPS-layer outputs
    (include_input_in_jk=False).  With JK='concat' and num_gnn_layers=4,
    h_dim=512, the post-pool vector is 4 * 512 = 2048 before the shared FFN.
    """

    def __init__(
        self,
        num_gnn_layers: int = 4,
        graph_pooling: str = "attention",
        JK: str = "concat",
        node_dim: int = 52,
        edge_dim: int = 14,
        h_dim: int = 512,
        ffn_dim: int = 256,
        drop_ratio: float = 0.2,
        heads: int = 4,
        attn_type: str = "multihead",
        task_heads=None,
    ):
        super(GPS_net, self).__init__()

        if task_heads is None:
            self.task_heads = {"y_sol": 1, "y_logd": 1, "y_clint": 1}
        else:
            self.task_heads = task_heads

        self.num_gnn_layers = num_gnn_layers
        self.JK = JK
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.ffn_dim = ffn_dim
        self.drop_ratio = drop_ratio

        # ── Input projections ────────────────────────────────────────────────
        # Project raw node/edge features to h_dim once; all GPS layers then
        # operate uniformly in h_dim.  LayerNorm is preferred over BatchNorm
        # here: it is batch-size independent and behaves identically at train
        # and inference time, which is important for Transformer-style models.
        self.node_proj = Linear(node_dim, h_dim)
        self.node_norm = LayerNorm(h_dim)
        self.edge_proj = Linear(edge_dim, h_dim)
        self.edge_norm = LayerNorm(h_dim)

        # ── GPS layers ────────────────────────────────────────────────────────
        # Each GPSConv wraps a local GINEConv (edge_dim=h_dim after projection)
        # and a global multi-head self-attention.
        self.convs = ModuleList()
        for _ in range(num_gnn_layers):
            gine_nn = Sequential(
                Linear(h_dim, h_dim),
                ReLU(),
                Linear(h_dim, h_dim),
            )
            conv = GPSConv(
                channels=h_dim,
                conv=GINEConv(gine_nn, edge_dim=h_dim),
                heads=heads,
                attn_type=attn_type,
            )
            self.convs.append(conv)

        # Pooling
        # JK-concat dim: only GPS-layer outputs
        if JK == "concat":
            pool_dim = num_gnn_layers * h_dim
        else:
            pool_dim = h_dim

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=Linear(pool_dim, 1))

        # Shared FFN
        if JK == "concat":
            self.shared_ffn = Sequential(
                Linear(pool_dim, ffn_dim * 4),
                ReLU(),
                Dropout(0.3),
                Linear(ffn_dim * 4, ffn_dim),
                ReLU(),
                Dropout(0.2),
            )
        else:
            self.shared_ffn = Sequential(
                Linear(pool_dim, ffn_dim * 2),
                ReLU(),
                Dropout(0.3),
                Linear(ffn_dim * 2, ffn_dim),
                ReLU(),
                Dropout(0.2),
            )

        # Task-specific heads
        self.task_predictors = nn.ModuleDict()
        for task_name, output_dim in self.task_heads.items():
            self._create_task_head(task_name, output_dim)

        # Current active task (None = pretrain / all tasks active)
        self.active_task = None

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # Project inputs to h_dim
        x = F.relu(self.node_norm(self.node_proj(x)))  # [N, h_dim]
        edge_attr = F.relu(self.edge_norm(self.edge_proj(edge_attr)))  # [E, h_dim]

        # GPS layers — collect outputs for JK (input representation excluded)
        h_list = []
        h = x
        for layer in range(self.num_gnn_layers):
            h = self.convs[layer](h, edge_index, batch, edge_attr=edge_attr)
            if layer < self.num_gnn_layers - 1:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            else:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        # Jumping knowledge aggregation
        if self.JK == "concat":
            node_out = torch.cat(h_list, dim=1)  # [N, num_layers * h_dim]
        elif self.JK == "last":
            node_out = h_list[-1]
        elif self.JK == "max":
            node_out = torch.stack(h_list, dim=0).max(dim=0)[0]
        elif self.JK == "sum":
            node_out = torch.stack(h_list, dim=0).sum(dim=0)

        # Graph-level pooling
        graph_out = self.pool(node_out, batch)  # [B, pool_dim]
        shared = self.shared_ffn(graph_out)  # [B, ffn_dim]

        # Task-specific predictions
        if self.active_task is not None:
            return self.task_predictors[self.active_task](shared)

        outputs = {}
        for task_name in self.task_heads:
            outputs[task_name] = self.task_predictors[task_name](shared)
        return outputs

    # Task-head helpers
    def _create_task_head(self, task_name, output_dim=1):
        """Create a task-specific prediction head."""
        self.task_predictors[task_name] = nn.Sequential(
            nn.Linear(self.ffn_dim, self.ffn_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.ffn_dim // 4, output_dim),
        )
        self.task_heads[task_name] = output_dim

    def add_task(self, task_name, output_dim=1):
        """Add a new task head to the model."""
        if task_name in self.task_heads:
            print(f"Task {task_name} already exists. Skipping.")
            return
        self._create_task_head(task_name, output_dim)
        print(f"Added new task head: {task_name}")

    def set_fine_tuning_mode(self, task_name=None):
        """Set fine-tuning mode (identical semantics to GNN_net).

        task_name=None  →  pretrain mode, all parameters trainable.
        task_name=<str> →  only the specified task head gradients are active;
                           backbone (convs) and shared_ffn remain trainable.
        """
        if task_name is not None and task_name not in self.task_heads:
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Available tasks: {list(self.task_heads.keys())}. "
                f"Use add_task('{task_name}') to add a new task before fine-tuning."
            )
        self.active_task = task_name

        if task_name is not None:
            # Backbone and shared FFN remain trainable
            for param in self.node_proj.parameters():
                param.requires_grad = True
            for param in self.node_norm.parameters():
                param.requires_grad = True
            for param in self.edge_proj.parameters():
                param.requires_grad = True
            for param in self.edge_norm.parameters():
                param.requires_grad = True
            for param in self.convs.parameters():
                param.requires_grad = True
            for param in self.shared_ffn.parameters():
                param.requires_grad = True

            # Activate only the target task head
            for name, param in self.task_predictors.named_parameters():
                param.requires_grad = task_name in name
        else:
            # Pretrain mode — all parameters trainable
            for param in self.parameters():
                param.requires_grad = True
