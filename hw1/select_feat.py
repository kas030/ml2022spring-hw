feat_discard = [0]

def select_feat(data, test=False, select_all=True):
    """Selects useful features to perform regression"""
    if not test:
        x = data[:, :-1]
        y = data[:, -1]
    else:
        x = data
        y = []

    feat_idx = list(range(x.shape[1]))
    if not select_all:
        for col in feat_discard:
            feat_idx.remove(col)

    return x[:, feat_idx], y
