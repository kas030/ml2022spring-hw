import csv

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_model import MyModel
from covid_19_dataset import COVID19Dataset
from select_feat import select_feat
from config import config


def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def save_pred(preds, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tested_positive'])
        for id, test_positive in enumerate(preds):
            writer.writerow([id, test_positive])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(
        './models/model.ckpt',
        map_location=device,
        weights_only=True
    )
    state_dict = checkpoint['model_state_dict']
    feature_indices = checkpoint['feature_indices'].cpu().numpy()
    input_dim = state_dict["layers.0.weight"].shape[1]

    model = MyModel(input_dim).to(device)
    model.load_state_dict(state_dict)

    test_data = pd.read_csv('./data/covid.test.csv').values
    print(f"Test_data size: {test_data.shape}")

    x_test, _ = select_feat(test_data, test=True, select_all=config['select_all'])
    x_test = x_test[:, feature_indices]
    test_dataset = COVID19Dataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    preds = predict(test_loader, model, device)
    save_pred(preds, './data/covid.pred.csv')
