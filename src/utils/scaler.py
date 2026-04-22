from sklearn.preprocessing import MinMaxScaler


class DataScaler:
    """Handles normalization and inverse transformation."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)