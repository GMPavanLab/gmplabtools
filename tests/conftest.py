import pytest
import inspect
import functools


@pytest.fixture
def input_dict():
    return {
        "d": 3,
        "fpoints": 0.02,
        "ngrid": 1000,
        "qs": 1,
        "o": "pamm",
        "trajectory": "allpca.pca",
    }


class InputDictFixture:

    FIXTURE_NAME = 'input_dict'

    def __init__(self, **kwargs):
        self.__dict__ = dict(**kwargs)

    def __call__(self, func):

        @functools.wraps(func)
        def wraps(*args, **kwargs):

            new_args = list(args)
            new_kwargs = kwargs.copy()

            # update args
            func_args = list(inspect.signature(func).parameters.keys())
            arg_position = InputDictFixture.get_arg_position(
                func_args,
                InputDictFixture.FIXTURE_NAME
            )
            if arg_position is not None:
                try:
                    new_args[arg_position].update(self.__dict__)
                except IndexError:
                    pass

            # update kwargs
            try:
                new_kwargs[InputDictFixture.FIXTURE_NAME].update(self.__dict__)

            except KeyError:
                pass

            result = func(*new_args, **new_kwargs)
            return result

        return wraps

    @staticmethod
    def get_arg_position(arg_list, arg):
        if arg not in arg_list:
            return None
        for i, a in enumerate(arg_list):
            if a == arg:
                return i
        return None