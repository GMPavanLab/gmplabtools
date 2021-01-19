from sklearn.base import TransformerMixin


__all__ = ["NullTransformer"]


class NullTransformer(TransformerMixin):

    def fit(self, x):
        return x

    def predict(self, x):
        return x