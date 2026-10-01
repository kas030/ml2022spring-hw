from sklearn.feature_selection import SelectKBest, f_regression


feat_discard = [0]


def select_feat(data, test=False, select_all=True, selector=None):
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

    x = x[:, feat_idx]
    if selector is not None:
        x = selector.transform(x)

    return x, y


def fit_feature_selector(data, num_features):
    """Fit feature selection on training data only to prevent data leakage."""
    x, y = select_feat(data, test=False, select_all=False)
    selector = SelectKBest(score_func=f_regression, k=num_features)
    selector.fit(x, y)
    return selector
