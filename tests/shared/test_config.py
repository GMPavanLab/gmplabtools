import pytest

from gmplabtools.shared.config import _importer, import_external


def test_importer():
    # when:
    module = "gmplabtools.pamm"
    obj_name = "Pamm"

    # then:
    result = _importer(module, obj_name)
    assert result


def test_importer_fail():
    # when:
    module = "gmplabtools.pammmmmmm"
    obj_name = "Pamm"

    # then:
    result = _importer(module, obj_name)
    assert result is None


def test_import_external(config, ):
    # when:
    class_name = "NullTransformer"

    # then:
    result = import_external(config, class_name)
    assert result


def test_import_external_from_processing(config):
    # when:
    class_name = "NullTransformer"

    # then:
    result = import_external(config, class_name)
    assert result


def test_import_external_from_installed(config):
    # when:
    class_name = "array"

    # then:
    result = import_external(config, class_name)
    assert result


def test_import_external_from_file(config):
    # when:
    class_name = "Myclass"

    # then:
    config
    obj = import_external(config, class_name)
    assert obj


def test_import_external_fail(config):
    # when:
    class_name = "Myclassssss"

    # then:
    with pytest.raises(ModuleNotFoundError):
        _ = import_external(config, class_name)


def test_config_recursive(config):
    assert config.transform
    assert isinstance(config.pymodule, str)
    assert isinstance(config.transform["soap_param"], dict)

