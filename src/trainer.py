import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def compute_sample_weights_classification(dataset, label_attr='y', target_ratio = 0.25):
    """
    Compute sample weights for classification tasks to handle class imbalance.
    Args:
        dataset: PyTorch Geometric dataset
        label_attr: Name of the label attribute in the dataset (default: 'y')
        minority_weight: Weight multiplier for minority class (default: 4.0 for 80/20 split)
                        Use 4.0 for 20% minority/80% majority balance in mini-batches
    Returns:
        List of sample weights for each sample in the dataset
    """
    labels = []
    for i in range(len(dataset)):
        data = dataset[i]
        label = getattr(data, label_attr).item()
        labels.append(label)

    labels = np.array(labels)

    # Count positive and negative samples
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    print(f"Class distribution: {n_positive} positive ({n_positive/(n_positive+n_negative)*100:.1f}%), "
          f"{n_negative} negative ({n_negative/(n_positive+n_negative)*100:.1f}%)")

    # Assign weights: minority class gets higher weight
    sample_weights = np.ones(len(labels))

    #target_ratio = 0.25 # 1:4 ratio
    pos_weight = (n_negative / n_positive) * target_ratio
    sample_weights[labels == 1] = pos_weight
    print(f"Minority class (positive) weight: {pos_weight:.2f}, Majority class weight: 1.0")

    return sample_weights.tolist()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of the true class
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class MultitaskTrainer:
    """
    L1Loss(reduction='none') for custom masking (multitask pretraining)
    L1Loss(reduction='mean') for single-task fine-tuning
    """
    def __init__(self, model, tasks=['y_sol', 'y_logd', 'y_hlm', 'y_mlm', 'y_mdck'], mode = 'pretrain', device='cuda'):
        self.model = model
        self.tasks = tasks
        self.device = device
        self.model.to(device)

        if mode == 'pretrain':
            self.criterion = nn.L1Loss(reduction='none')      # 'none' for custom masking
        elif mode == 'finetune':
            self.criterion = nn.L1Loss()

    def pretrain(self, train_dataset, test_dataset, epochs = 200, lr = 1e-4, task_weights=None):
        """
        Pretrain the model on all tasks simultaneously.

        Args:
            train_dataset: The training dataset
            test_dataset: The test dataset
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            task_weights: Optional dictionary of weights for each task's loss
        """
        self.model.set_fine_tuning_mode(None)

        if task_weights is None:
            task_weights = {task: 1.0 for task in self.tasks}

        train_loader = DataLoader(train_dataset, batch_size = 8192, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = 4096)

        optimizer = optim.AdamW(self.model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        best_loss = float('inf')

        for epoch in range(epochs):

            self.model.train()
            train_losses = {task: 0.0 for task in self.tasks}
            train_counts = {task: 0 for task in self.tasks}
            total_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(data)

                # Calculate loss for each task and weighted sum
                batch_loss = 0.0
                for task in self.tasks:
                    target = getattr(data, task)
                    mask = ~torch.isnan(target)         # non-NaN values
                    if not mask.any():
                        continue

                    task_losses = self.criterion(outputs[task].squeeze()[mask], target[mask])
                    # Reduce to scalar and apply task weight
                    task_loss = task_losses.mean() * task_weights[task]
                    batch_loss += task_loss

                    train_losses[task] += task_loss.item() * mask.sum().item()
                    train_counts[task] += mask.sum().item()

                if batch_loss > 0:                      # Only backward if we have a valid loss
                    total_loss += batch_loss.item()
                    batch_loss.backward()
                    optimizer.step()
                    scheduler.step()

            # Calculate average losses
            for task in self.tasks:
                if train_counts[task] > 0:
                    train_losses[task] /= train_counts[task]
                else:
                    train_losses[task] = float('nan')

            # Evaluation
            self.model.eval()
            test_losses = {task: 0.0 for task in self.tasks}
            test_counts = {task: 0 for task in self.tasks}
            test_total_loss = 0.0

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)

                    batch_loss = 0.0
                    for task in self.tasks:
                        target = getattr(data, task)
                        mask = ~torch.isnan(target)
                        if not mask.any():
                            continue

                        task_losses = self.criterion(outputs[task].squeeze()[mask], target[mask])

                        task_loss = task_losses.mean() * task_weights[task]
                        batch_loss += task_loss

                        test_losses[task] += task_loss.item() * mask.sum().item()
                        test_counts[task] += mask.sum().item()

                    if batch_loss > 0:
                        test_total_loss += batch_loss.item()

            for task in self.tasks:
                if test_counts[task] > 0:
                    test_losses[task] /= test_counts[task]
                else:
                    test_losses[task] = float('nan')


            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {total_loss:.4f} | ' +
                  ' | '.join([f'{task}: {loss:.4f}' for task, loss in train_losses.items()]))
            print(f'  Test Loss: {test_total_loss:.4f} | ' +
                  ' | '.join([f'{task}: {loss:.4f}' for task, loss in test_losses.items()]))

            # Save best model
            if test_total_loss < best_loss:
                best_loss = test_total_loss
                torch.save(self.model.state_dict(), 'best_pretrained_Novartis_full_data.pt')
                print('  Model saved.')

        return self.model


    def finetune(self, task_name, train_dataset, test_dataset, epochs = 50, lr = 1e-4):
        """
        Fine-tune the model for a specific task.

        Args:
            task_name: The name of the task to fine-tune on
            train_dataset: The training dataset
            test_dataset: The test dataset
            batch_size: Batch size for training (smaller for fine-tuning)
            epochs: Number of training epochs
            lr: Learning rate (lower for fine-tuning)
        """
        self.model.set_fine_tuning_mode(task_name)

        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 16)

        # Optimizer - only optimize the task-specific parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        best_loss = float('inf')
        smallest_difference = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)                                           # Now returns only the task-specific output
                loss = self.criterion(output.squeeze(), getattr(data, 'y'))         # task_name

                train_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Calculate average loss
            train_loss /= len(train_dataset)

            # Evaluation
            self.model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), getattr(data, 'y'))    # task_name
                    test_loss += loss.item() * data.num_graphs

            test_loss /= len(test_dataset)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(self.model.state_dict(), f'best_finetuned_model_{task_name}_Novartis.pt')
                print('  Model saved.')

        return self.model

    def finetune_classification(self, task_name, train_dataset, test_dataset, epochs=50, lr=1e-4,
                                pos_weight=None, loss_type='bce', alpha=None, gamma=2.0,
                                use_weighted_sampler=False, target_ratio = 0.25):
        """
        Fine-tune the model for a binary classification task using BCEWithLogitsLoss.

        Args:
            task_name: The name of the classification task to fine-tune on
            train_dataset: The training dataset
            test_dataset: The test dataset
            epochs: Number of training epochs
            lr: Learning rate (lower for fine-tuning)
            pos_weight: Weight for positive class (torch.Tensor). Only used with BCEWithLogitsLoss.
                       If None, no weighting is applied. Should be ratio of negative/positive samples.
            loss_type: 'bce' for BCEWithLogitsLoss or 'focal' for FocalLoss. Default: 'bce'
            alpha: Alpha parameter for FocalLoss (0-1). Only used when loss_type='focal'.
                   Weighting factor for positive class. Default: None
            gamma: Gamma parameter for FocalLoss. Only used when loss_type='focal'.
                   Focusing parameter for hard examples. Default: 2.0
            use_weighted_sampler: Whether to use WeightedRandomSampler for balanced batches
            minority_weight: Weight for minority class in sampler (default: 4.0 for 20/80 balance)
        """
        self.model.set_fine_tuning_mode(task_name)

        if use_weighted_sampler:
            print("Using WeightedRandomSampler for balanced mini-batches...")
            sample_weights = compute_sample_weights_classification(train_dataset, label_attr='y', target_ratio = target_ratio)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=2048, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=512)

        # classification losses
        if loss_type == 'focal':
            criterion_cls = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
            print(f"Using FocalLoss with alpha={alpha}, gamma={gamma}")
        else:
            if pos_weight is not None:
                pos_weight = pos_weight.to(self.device)
            criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight}")

        # Optimizer - only optimize the task-specific parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=lr)       # default weight_decay=1e-2

        best_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)  # Returns logits for the classification task
                loss = criterion_cls(output.squeeze(), getattr(data, 'y').float())

                train_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()

            train_loss /= len(train_dataset)

            # Evaluation
            self.model.eval()
            test_loss = 0.0
            all_preds = []
            all_labels = []
            all_logits = []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    loss = criterion_cls(output.squeeze(), getattr(data, 'y').float())

                    test_loss += loss.item() * data.num_graphs

                    # Collect predictions and labels for PR-AUC
                    logits = output.squeeze().cpu().numpy()
                    probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                    labels = getattr(data, 'y').cpu().numpy()
                    all_logits.extend(logits.tolist() if logits.ndim > 0 else [logits.item()])
                    all_preds.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
                    all_labels.extend(labels.tolist() if labels.ndim > 0 else [labels.item()])

            test_loss /= len(test_dataset)
            pr_auc = average_precision_score(all_labels, all_preds)
            roc_auc = roc_auc_score(all_labels, all_preds)

            # Logits distribution by class
            all_logits = torch.tensor(all_logits)
            all_labels = torch.tensor(all_labels)
            pos_logits = all_logits[all_labels == 1]
            neg_logits = all_logits[all_labels == 0]

            pos_mean = pos_logits.mean().item() if len(pos_logits) > 0 else 0.0
            pos_std = pos_logits.std().item() if len(pos_logits) > 0 else 0.0
            neg_mean = neg_logits.mean().item() if len(neg_logits) > 0 else 0.0
            neg_std = neg_logits.std().item() if len(neg_logits) > 0 else 0.0

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}')
            print(f'  Logits - Pos: {pos_mean:.3f}±{pos_std:.3f} | Neg: {neg_mean:.3f}±{neg_std:.3f}')

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.state_dict(), f'finetuned_models/best_finetuned_{task_name}.pt')
                print('  Model saved.')

        return self.model
    

    def cross_val(self, task_name, train_dataset, test_dataset, epochs = 50, lr = 1e-4):
        # Set model to fine-tuning mode for the specific task
        self.model.set_fine_tuning_mode(task_name)

        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 16)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        best_loss = float('inf')
        smallest_difference = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)  # Now returns only the task-specific output
                loss = self.criterion(output.squeeze(), getattr(data, 'y'))      # task_name

                train_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                scheduler.step()

            train_loss /= len(train_dataset)

        self.model.eval()
        preds = torch.Tensor()
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.cat((preds, output.squeeze().cpu().flatten()), 0)      # output[task_name]

        return preds