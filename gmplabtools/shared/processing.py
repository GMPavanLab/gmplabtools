from sklearn.base import TransformerMixin
import sklearn.decomposition as dim_red

__all__ = ["NullTransformer"] + [cls for cls in dir(dim_red) if hasattr(cls, "transform")]


class NullTransformer(TransformerMixin):

    def fit(self, x):
        return x

    def predict(self, x):
        return x
