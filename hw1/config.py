config = {
    'seed': 5201314,
    'select_all': False,  # Whether to use all features.
    'num_features': 34,
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 1000,     # Number of epochs.
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-6,
    'early_stop': 120,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'
}
