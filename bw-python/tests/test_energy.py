"""Tests for EnergyBuilder interpolation."""
import numpy as np
import pytest

from bw.energy import EnergyBuilder


@pytest.fixture
def builder():
    return EnergyBuilder()


@pytest.fixture
def simple_data():
    """Simple 2-point example: energy=[[100, 200]], time=[0, 10]."""
    return {"energy": [[100, 200]], "time": [0, 10]}


class TestShape:
    """All methods should produce (n_individuals, n_days + 1) output."""

    @pytest.mark.parametrize("method", sorted(EnergyBuilder.METHODS))
    def test_output_shape(self, builder, simple_data, method):
        np.random.seed(42)
        result = builder.build(**simple_data, method=method)
        assert result.shape == (1, 11)

    @pytest.mark.parametrize("method", sorted(EnergyBuilder.METHODS))
    def test_multi_individual_shape(self, builder, method):
        energy = [[100, 200], [300, 400], [500, 600]]
        time = [0, 10]
        np.random.seed(42)
        result = builder.build(energy, time, method=method)
        assert result.shape == (3, 11)


class TestLinear:
    """Linear interpolation should give exact endpoints and linear values."""

    def test_endpoints(self, builder, simple_data):
        result = builder.build(**simple_data, method="linear")
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, -1] == pytest.approx(200.0)

    def test_midpoint(self, builder, simple_data):
        result = builder.build(**simple_data, method="linear")
        assert result[0, 5] == pytest.approx(150.0)

    def test_all_values_linear(self, builder, simple_data):
        result = builder.build(**simple_data, method="linear")
        expected = np.linspace(100, 200, 11)
        np.testing.assert_allclose(result[0], expected)


class TestStepwise:
    """Stepwise methods should hold constant values within segments."""

    def test_stepwise_l_holds_left_value(self, builder, simple_data):
        result = builder.build(**simple_data, method="stepwise_l")
        # All interior points should equal the left endpoint
        for i in range(10):
            assert result[0, i] == pytest.approx(100.0)
        # Last day = last measurement
        assert result[0, -1] == pytest.approx(200.0)

    def test_stepwise_r_holds_right_value(self, builder, simple_data):
        result = builder.build(**simple_data, method="stepwise_r")
        # All interior points should equal the right endpoint
        for i in range(10):
            assert result[0, i] == pytest.approx(200.0)
        # Last day = last measurement
        assert result[0, -1] == pytest.approx(200.0)

    def test_stepwise_l_multi_segment(self, builder):
        energy = [[100, 200, 300]]
        time = [0, 5, 10]
        result = builder.build(energy, time, method="stepwise_l")
        # First segment: days 0-4 should be 100
        for i in range(5):
            assert result[0, i] == pytest.approx(100.0)
        # Second segment: days 5-9 should be 200
        for i in range(5, 10):
            assert result[0, i] == pytest.approx(200.0)
        # Last day
        assert result[0, -1] == pytest.approx(300.0)

    def test_stepwise_r_multi_segment(self, builder):
        energy = [[100, 200, 300]]
        time = [0, 5, 10]
        result = builder.build(energy, time, method="stepwise_r")
        # First segment: days 0-4 should be 200
        for i in range(5):
            assert result[0, i] == pytest.approx(200.0)
        # Second segment: days 5-9 should be 300
        for i in range(5, 10):
            assert result[0, i] == pytest.approx(300.0)
        # Last day
        assert result[0, -1] == pytest.approx(300.0)


class TestBrownian:
    """Brownian bridge should match endpoints exactly."""

    def test_endpoints(self, builder, simple_data):
        np.random.seed(42)
        result = builder.build(**simple_data, method="brownian")
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, -1] == pytest.approx(200.0)

    def test_reproducible_with_seed(self, builder, simple_data):
        np.random.seed(123)
        r1 = builder.build(**simple_data, method="brownian")
        np.random.seed(123)
        r2 = builder.build(**simple_data, method="brownian")
        np.testing.assert_array_equal(r1, r2)

    def test_multi_segment_endpoints(self, builder):
        energy = [[100, 200, 300]]
        time = [0, 5, 10]
        np.random.seed(42)
        result = builder.build(energy, time, method="brownian")
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, 5] == pytest.approx(200.0)
        assert result[0, -1] == pytest.approx(300.0)


class TestExponentialLogarithmic:
    """Exponential and logarithmic methods should match endpoints."""

    def test_exponential_endpoints(self, builder, simple_data):
        result = builder.build(**simple_data, method="exponential")
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, -1] == pytest.approx(200.0)

    def test_logarithmic_endpoints(self, builder, simple_data):
        result = builder.build(**simple_data, method="logarithmic")
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, -1] == pytest.approx(200.0)


class TestValidation:
    """Validation should raise errors for bad inputs."""

    def test_bad_method(self, builder, simple_data):
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            builder.build(**simple_data, method="cubic")

    def test_time_not_starting_at_zero(self, builder):
        with pytest.raises(ValueError, match="time must start at 0"):
            builder.build([[100, 200]], [1, 10], method="linear")

    def test_time_too_short(self, builder):
        with pytest.raises(ValueError, match="time must have at least 2 elements"):
            builder.build([[100]], [0], method="linear")

    def test_time_not_increasing(self, builder):
        with pytest.raises(ValueError, match="time must be strictly increasing"):
            builder.build([[100, 200, 300]], [0, 5, 3], method="linear")

    def test_dimension_mismatch(self, builder):
        with pytest.raises(ValueError, match="energy has 3 columns but time has 2"):
            builder.build([[100, 200, 300]], [0, 10], method="linear")
