

from unittest import TestCase
from architecture.tuningfeatures import TuningParam
from architecture.tuningfeatures import TuningFeatures


class TestTuningFeatures(TestCase):
    def test_tuning_param_not_normalize_real(self):
        tuning_param = TuningParam('f1', 'real', False, 0.0, 1.0)
        assert tuning_param.lower_bound == 0.0, 'test1 failed'
        print(str(tuning_param.param_value))
        new_value = 0.6
        tuning_param.update(new_value)
        assert tuning_param() == new_value

    def test_tuning_param_normalize_real(self):
        tuning_param = TuningParam('f1', 'ordinal', False, -2.0, 8.0)
        assert tuning_param.lower_bound == -2.0, 'test1 failed'
        print(str(tuning_param.param_value))
        new_value = 3
        tuning_param.update(new_value)
        # res = tuning_param()
        assert tuning_param() == 3

    def test_tuning_param_normalize_ordinal(self):
        tuning_param = TuningParam('f1', 'ordinal', True, -2.0, 8.0)
        assert tuning_param.lower_bound == -2.0, 'test1 failed'

        print(str(tuning_param.param_value))
        tuning_param.param_value = 3
        res = tuning_param()
        assert res == 0.5, 'TuningParam.call() failed'
        new_value = 0.52
        tuning_param.update(new_value)

        assert tuning_param.param_value == 3
        new_value = 0.62
        tuning_param.update(new_value)
        assert tuning_param.param_value == 4

    def test_tuning_features_init(self):
        tuning_features = TuningFeatures()
        param_1 = TuningParam('F1', 'real', False, 0.0, 1.0)
        param_2 = TuningParam('F4', 'real', True, -2.0, 8.0)
        param_3 = TuningParam('F3', 'ordinal', False, 0.0, 8.0)
        param_4 = TuningParam('F2', 'ordinal', True, 0.0, 8.0)

        tuning_features.add_param(param_1)
        tuning_features.add_param(param_2)
        tuning_features.add_param(param_3)
        tuning_features.add_param(param_4)

        print(f'Parameters\n: {str(tuning_features)}')
        print(f'Parameter values: {tuning_features.get_values()}')
        tuning_features.sort()
        print(f'After ordering parameters\n: {str(tuning_features)}')
        print(f'After ordering parameter values: {tuning_features.get_values()}')

        cat_value_2 = tuning_features.get_param('F2')
        assert cat_value_2.param_name == 'F2', 'test1 for tuning features failed'
        print(str(cat_value_2))
        try:
            cat_value_3 = tuning_features.get_param('G3')
        except KeyError as e:
            print(f'Succeeded test1 for undefined key {str(e)}')


    def test_tuning_features_update(self):
        import numpy
        tuning_features = TuningFeatures()
        param_1 = TuningParam('F1', 'real', False, 0.0, 1.0)
        param_2 = TuningParam('F4', 'real', True, -2.0, 8.0)
        param_3 = TuningParam('F3', 'ordinal', True, 0.0, 8.0)
        param_4 = TuningParam('F2', 'ordinal', True, 0.0, 8.0)

        tuning_features.add_param(param_1)
        tuning_features.add_param(param_2)
        tuning_features.add_param(param_3)
        tuning_features.add_param(param_4)

        tuning_features.sort()
        new_weights = numpy.array([0.6, 0.2, 0.6, 0.9], dtype=numpy.float32)
        tuning_features.update(new_weights)

