"""Tests for child weight dynamics model."""
import math
import numpy as np
import pytest

from bw.child_model import child_reference_FFM_FM, child_weight, Child
from bw.constants import FFM_REF_MALE, FFM_REF_FEMALE, FM_REF_MALE, FM_REF_FEMALE


# ---------------------------------------------------------------------------
# Reference FFM/FM tests
# ---------------------------------------------------------------------------

class TestChildReferenceFFMFM:
    """Test child_reference_FFM_FM at known ages."""

    def test_male_age_2(self):
        ref = child_reference_FFM_FM(2.0, 0)
        # Age 2 -> index 0
        assert ref["FFM"][0] == pytest.approx(FFM_REF_MALE[0])
        assert ref["FM"][0] == pytest.approx(FM_REF_MALE[0])

    def test_female_age_2(self):
        ref = child_reference_FFM_FM(2.0, 1)
        assert ref["FFM"][0] == pytest.approx(FFM_REF_FEMALE[0])
        assert ref["FM"][0] == pytest.approx(FM_REF_FEMALE[0])

    def test_male_age_10(self):
        ref = child_reference_FFM_FM(10.0, 0)
        # Age 10 -> index 8
        assert ref["FFM"][0] == pytest.approx(FFM_REF_MALE[8])
        assert ref["FM"][0] == pytest.approx(FM_REF_MALE[8])

    def test_female_age_10(self):
        ref = child_reference_FFM_FM(10.0, 1)
        assert ref["FFM"][0] == pytest.approx(FFM_REF_FEMALE[8])
        assert ref["FM"][0] == pytest.approx(FM_REF_FEMALE[8])

    def test_male_age_18(self):
        ref = child_reference_FFM_FM(18.0, 0)
        # Age >= 18 -> index 16 (last entry)
        assert ref["FFM"][0] == pytest.approx(FFM_REF_MALE[16])
        assert ref["FM"][0] == pytest.approx(FM_REF_MALE[16])

    def test_female_age_18(self):
        ref = child_reference_FFM_FM(18.0, 1)
        assert ref["FFM"][0] == pytest.approx(FFM_REF_FEMALE[16])
        assert ref["FM"][0] == pytest.approx(FM_REF_FEMALE[16])

    def test_interpolation_midpoint(self):
        """Age 5.5 should interpolate between index 3 and 4."""
        ref = child_reference_FFM_FM(5.5, 0)
        expected_ffm = FFM_REF_MALE[3] + 0.5 * (FFM_REF_MALE[4] - FFM_REF_MALE[3])
        assert ref["FFM"][0] == pytest.approx(expected_ffm)

    def test_vectorized(self):
        ref = child_reference_FFM_FM([6.0, 10.0], [0, 1])
        assert ref["FFM"].shape == (2,)
        assert ref["FM"].shape == (2,)
        # First is male age 6 -> index 4
        assert ref["FFM"][0] == pytest.approx(FFM_REF_MALE[4])
        # Second is female age 10 -> index 8
        assert ref["FFM"][1] == pytest.approx(FFM_REF_FEMALE[8])


# ---------------------------------------------------------------------------
# child_weight integration tests
# ---------------------------------------------------------------------------

class TestChildWeightMale:
    """Test child_weight for a male child age 6."""

    def test_bw_increases(self):
        result = child_weight(6.0, "male", days=365, dt=1.0)
        bw = result["Body_Weight"]
        assert bw.shape[0] == 1  # 1 individual
        assert bw.shape[1] == 366  # 365 days + initial
        # Body weight should increase over a year for a growing child
        assert bw[0, -1] > bw[0, 0]

    def test_output_keys(self):
        result = child_weight(6.0, "male", days=30)
        assert "Time" in result
        assert "Age" in result
        assert "Fat_Free_Mass" in result
        assert "Fat_Mass" in result
        assert "Body_Weight" in result
        assert result["Model_Type"] == "Children"
        assert result["Correct_Values"] is True

    def test_initial_values_match_reference(self):
        result = child_weight(6.0, "male", days=10)
        ref = child_reference_FFM_FM(6.0, 0)
        assert result["Fat_Free_Mass"][0, 0] == pytest.approx(ref["FFM"][0])
        assert result["Fat_Mass"][0, 0] == pytest.approx(ref["FM"][0])

    def test_bw_equals_ffm_plus_fm(self):
        result = child_weight(6.0, "male", days=100)
        np.testing.assert_allclose(
            result["Body_Weight"],
            result["Fat_Free_Mass"] + result["Fat_Mass"],
            atol=1e-10,
        )

    def test_age_advances(self):
        result = child_weight(6.0, "male", days=365)
        # Age should advance by 365/365 = 1 year
        assert result["Age"][0, -1] == pytest.approx(7.0, abs=0.01)


class TestChildWeightFemale:
    """Test child_weight for a female child age 10."""

    def test_bw_increases(self):
        result = child_weight(10.0, "female", days=365)
        bw = result["Body_Weight"]
        assert bw[0, -1] > bw[0, 0]

    def test_initial_values_match_reference(self):
        result = child_weight(10.0, "female", days=10)
        ref = child_reference_FFM_FM(10.0, 1)
        assert result["Fat_Free_Mass"][0, 0] == pytest.approx(ref["FFM"][0])
        assert result["Fat_Mass"][0, 0] == pytest.approx(ref["FM"][0])


# ---------------------------------------------------------------------------
# Richardson logistic tests
# ---------------------------------------------------------------------------

class TestRichardsonParams:
    """Test child_weight with Richardson generalized logistic intake."""

    def test_richardson_runs(self):
        rp = {"K": 2000.0, "Q": 1.0, "A": 1000.0, "B": 0.5, "nu": 1.0, "C": 1.0}
        result = child_weight(6.0, "male", days=100, richardson_params=rp)
        bw = result["Body_Weight"]
        assert bw.shape == (1, 101)
        # Should produce finite values
        assert np.all(np.isfinite(bw))

    def test_richardson_bw_positive(self):
        rp = {"K": 1800.0, "Q": 1.0, "A": 1200.0, "B": 0.3, "nu": 1.0, "C": 1.0}
        result = child_weight(8.0, "female", days=200, richardson_params=rp)
        assert np.all(result["Body_Weight"] > 0)


# ---------------------------------------------------------------------------
# Vectorized (multiple individuals) tests
# ---------------------------------------------------------------------------

class TestVectorized:
    """Test child_weight with multiple individuals."""

    def test_two_individuals(self):
        result = child_weight([6.0, 10.0], ["male", "female"], days=100)
        assert result["Body_Weight"].shape == (2, 101)
        assert result["Age"].shape == (2, 101)
        # Both should grow
        assert result["Body_Weight"][0, -1] > result["Body_Weight"][0, 0]
        assert result["Body_Weight"][1, -1] > result["Body_Weight"][1, 0]

    def test_three_males(self):
        result = child_weight([4.0, 8.0, 12.0], ["male", "male", "male"], days=50)
        assert result["Body_Weight"].shape == (3, 51)
        # Older children should have higher initial body weight
        assert result["Body_Weight"][2, 0] > result["Body_Weight"][1, 0]
        assert result["Body_Weight"][1, 0] > result["Body_Weight"][0, 0]

    def test_numeric_sex(self):
        """Test that numeric sex codes (0/1) work."""
        result = child_weight([6.0, 10.0], [0, 1], days=30)
        assert result["Body_Weight"].shape == (2, 31)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_age_too_low(self):
        with pytest.raises(ValueError, match="age must be between 2 and 18"):
            child_weight(1.0, "male", days=10)

    def test_age_too_high(self):
        with pytest.raises(ValueError, match="age must be between 2 and 18"):
            child_weight(19.0, "male", days=10)

    def test_invalid_sex_string(self):
        with pytest.raises(ValueError, match="sex must be 'male' or 'female'"):
            child_weight(6.0, "unknown", days=10)

    def test_invalid_sex_numeric(self):
        with pytest.raises(ValueError, match="numeric sex must be 0 or 1"):
            child_weight(6.0, [2], days=10)

    def test_mismatched_fm_length(self):
        with pytest.raises(ValueError, match="FM must have the same length"):
            child_weight(6.0, "male", FM=[1.0, 2.0], days=10)

    def test_mismatched_ffm_length(self):
        with pytest.raises(ValueError, match="FFM must have the same length"):
            child_weight(6.0, "male", FFM=[1.0, 2.0], days=10)


# ---------------------------------------------------------------------------
# dt parameter tests
# ---------------------------------------------------------------------------

class TestDtParameter:
    """Test that different dt values produce consistent results."""

    def test_half_day_step(self):
        result = child_weight(6.0, "male", days=30, dt=0.5)
        # With dt=0.5, nsims = 60, so shape is (1, 61)
        assert result["Body_Weight"].shape == (1, 61)
        assert np.all(np.isfinite(result["Body_Weight"]))
