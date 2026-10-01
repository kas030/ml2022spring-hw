import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from covid_19_dataset import COVID19Dataset
from my_model import MyModel
from select_feat import fit_feature_selector, select_feat
from config import config


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)





def trainer(train_loader, valid_loader, model, feature_indices, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define the loss function.

    # Define the optimization algorithm.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        total_train_loss = 0.0
        train_sample_count = 0

        # To visualize the training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move the data to device.

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()   # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            total_train_loss += loss.detach().item() * x.size(0)
            train_sample_count += x.size(0)

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = total_train_loss / train_sample_count
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # Validation
        model.eval()  # Set your model to evaluation mode.
        total_valid_loss = 0.0
        valid_sample_count = 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            total_valid_loss += loss.item() * x.size(0)
            valid_sample_count += x.size(0)

        mean_valid_loss = total_valid_loss / valid_sample_count
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_indices': torch.as_tensor(feature_indices, dtype=torch.long)
            }, config['save_path'])  # Save the best model and selected features.
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

if __name__ == '__main__':
    same_seed(config['seed'])

    train_data = pd.read_csv('./data/covid.train.csv').values

    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
    print(f"Train_data size: {train_data.shape}")
    print(f"Valid_data size: {valid_data.shape}")

    feature_selector = fit_feature_selector(train_data, config['num_features'])
    x_train, y_train = select_feat(
        train_data, test=False, select_all=config['select_all'], selector=feature_selector
    )
    x_valid, y_valid = select_feat(
        valid_data, test=False, select_all=config['select_all'], selector=feature_selector
    )

    print(f'number of features: {x_train.shape[1]}')

    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # Start training.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Start training...")
    model = MyModel(input_dim=x_train.shape[1]).to(device)
    trainer(
        train_loader,
        valid_loader,
        model,
        feature_selector.get_support(indices=True),
        config,
        device
    )
