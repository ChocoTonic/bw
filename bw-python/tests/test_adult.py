"""Tests for the adult body weight model."""

import math
import numpy as np
import pytest

from bw.adult_model import (
    Adult,
    adult_weight,
    estimate_ecf,
    estimate_fat,
    estimate_rmr,
)


# ---------------------------------------------------------------------------
# Tests for baseline estimation functions
# ---------------------------------------------------------------------------

class TestEstimateRMR:
    def test_male(self):
        # Mifflin-St Jeor: 5 + 9.99*bw + 625*ht - 4.92*age
        rmr = estimate_rmr(76.0, 1.73, 36.0, 0)
        expected = 5.0 + 9.99 * 76.0 + 625.0 * 1.73 - 4.92 * 36.0
        np.testing.assert_allclose(rmr, expected, rtol=1e-10)

    def test_female(self):
        # Mifflin-St Jeor: -161 + 9.99*bw + 625*ht - 4.92*age
        rmr = estimate_rmr(60.0, 1.60, 30.0, 1)
        expected = -161.0 + 9.99 * 60.0 + 625.0 * 1.60 - 4.92 * 30.0
        np.testing.assert_allclose(rmr, expected, rtol=1e-10)

    def test_vectorized(self):
        rmr = estimate_rmr([76, 60], [1.73, 1.60], [36, 30], [0, 1])
        assert rmr.shape == (2,)


class TestEstimateFat:
    def test_male(self):
        bw, ht, age = 76.0, 1.73, 36.0
        bmi = bw / ht**2
        expected = bw * (0.14 * age + 37.31 * math.log(bmi) - 103.94) / 100.0
        fat = estimate_fat(bw, ht, age, 0)
        np.testing.assert_allclose(fat, expected, rtol=1e-10)

    def test_female(self):
        bw, ht, age = 60.0, 1.60, 30.0
        bmi = bw / ht**2
        expected = bw * (0.14 * age + 39.96 * math.log(bmi) - 102.01) / 100.0
        fat = estimate_fat(bw, ht, age, 1)
        np.testing.assert_allclose(fat, expected, rtol=1e-10)


class TestEstimateECF:
    def test_male(self):
        ecf = estimate_ecf(76.0, 1.73, 36.0, 0)
        expected = 0.025 * 36.0 + 9.57 * 1.73 + 0.191 * 76.0 - 12.4
        np.testing.assert_allclose(ecf, expected, rtol=1e-10)

    def test_female(self):
        ecf = estimate_ecf(60.0, 1.60, 30.0, 1)
        expected = -4.0 + 5.98 * 1.60 + 0.167 * 60.0
        np.testing.assert_allclose(ecf, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for adult_weight
# ---------------------------------------------------------------------------

class TestAdultWeightMale:
    """Single male, 76 kg, 1.73 m, age 36, defaults, 365 days."""

    def test_baseline_stability(self):
        """With zero EIchange, body weight should remain stable."""
        result = adult_weight(76.0, 1.73, 36.0, "male", days=365)
        bw_series = result["Body_Weight"][:, 0]
        # Drift should be less than 0.1 kg from initial
        assert abs(bw_series[-1] - 76.0) < 0.1, (
            f"BW drifted to {bw_series[-1]:.4f} from 76.0"
        )

    def test_output_keys(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, days=10)
        expected_keys = {
            "Time", "Age", "Body_Weight", "Lean_Mass", "Fat_Mass",
            "Glycogen", "Extracellular_Fluid", "Adaptive_Thermogenesis",
            "BMI", "BMI_Category", "Energy_Intake", "Correct_Values",
            "Model_Type",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, days=10)
        assert result["Time"].shape == (11,)
        assert result["Body_Weight"].shape == (11, 1)
        assert result["Age"].shape == (11, 1)

    def test_initial_bw(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, days=10)
        np.testing.assert_allclose(result["Body_Weight"][0, 0], 76.0, rtol=1e-10)

    def test_model_type(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, days=10)
        assert result["Model_Type"] == "Adult"


class TestAdultWeightFemale:
    """Single female test."""

    def test_baseline_stability(self):
        result = adult_weight(60.0, 1.60, 30.0, "female", days=365)
        bw_series = result["Body_Weight"][:, 0]
        assert abs(bw_series[-1] - 60.0) < 0.1, (
            f"BW drifted to {bw_series[-1]:.4f} from 60.0"
        )

    def test_sex_string_conversion(self):
        result = adult_weight(60.0, 1.60, 30.0, "female", days=10)
        assert result["Body_Weight"].shape == (11, 1)


class TestWithEI:
    """Test with EI specified explicitly."""

    def test_custom_ei(self):
        # Provide a custom EI that differs from the steady state
        result = adult_weight(76.0, 1.73, 36.0, 0, EI=2500.0, days=30)
        assert result["Body_Weight"].shape == (31, 1)
        # With custom EI and zero EIchange, model should still run
        assert result["Correct_Values"] is True


class TestWithFat:
    """Test with fat specified explicitly."""

    def test_custom_fat(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, fat=15.0, days=30)
        assert result["Body_Weight"].shape == (31, 1)
        assert result["Correct_Values"] is True


class TestWithEIAndFat:
    """Test with both EI and fat specified."""

    def test_both_specified(self):
        result = adult_weight(76.0, 1.73, 36.0, 0, EI=2500.0, fat=15.0, days=30)
        assert result["Body_Weight"].shape == (31, 1)
        assert result["Correct_Values"] is True
        # Initial fat mass should match provided value
        np.testing.assert_allclose(result["Fat_Mass"][0, 0], 15.0, rtol=1e-6)


class TestInputValidation:
    def test_negative_bw(self):
        with pytest.raises(ValueError, match="Body weight must be positive"):
            adult_weight(-5.0, 1.73, 36.0, 0)

    def test_negative_height(self):
        with pytest.raises(ValueError, match="Height must be positive"):
            adult_weight(76.0, -1.73, 36.0, 0)

    def test_negative_age(self):
        with pytest.raises(ValueError, match="Age must be non-negative"):
            adult_weight(76.0, 1.73, -5.0, 0)

    def test_invalid_sex_string(self):
        with pytest.raises(ValueError, match="Unknown sex string"):
            adult_weight(76.0, 1.73, 36.0, "other")

    def test_invalid_sex_numeric(self):
        with pytest.raises(ValueError, match="sex must be 0"):
            adult_weight(76.0, 1.73, 36.0, 2)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            adult_weight([76.0, 80.0], [1.73], [36.0], [0])


class TestVectorized:
    """Test with multiple individuals."""

    def test_two_individuals(self):
        result = adult_weight(
            [76.0, 60.0], [1.73, 1.60], [36.0, 30.0], ["male", "female"],
            days=30,
        )
        assert result["Body_Weight"].shape == (31, 2)
        assert result["Fat_Mass"].shape == (31, 2)
        assert result["Lean_Mass"].shape == (31, 2)
        assert result["BMI"].shape == (31, 2)
        assert result["Time"].shape == (31,)
        assert len(result["BMI_Category"]) == 31
        assert len(result["BMI_Category"][0]) == 2

    def test_three_individuals_stability(self):
        bws = [70.0, 80.0, 90.0]
        result = adult_weight(
            bws, [1.70, 1.80, 1.85], [30.0, 40.0, 50.0], [0, 0, 0],
            days=365,
        )
        for j in range(3):
            drift = abs(result["Body_Weight"][-1, j] - bws[j])
            assert drift < 0.1, f"Individual {j}: BW drifted {drift:.4f} kg"


class TestBMICategories:
    def test_underweight(self):
        # BMI < 18.5: very light person
        result = adult_weight(45.0, 1.73, 25.0, 0, days=1)
        bmi = 45.0 / 1.73**2
        assert bmi < 18.5
        assert result["BMI_Category"][0][0] == "Underweight"

    def test_normal(self):
        # BMI 18.5-25
        result = adult_weight(70.0, 1.75, 30.0, 0, days=1)
        bmi = 70.0 / 1.75**2
        assert 18.5 <= bmi < 25.0
        assert result["BMI_Category"][0][0] == "Normal"

    def test_preobese(self):
        # BMI 25-30
        result = adult_weight(85.0, 1.73, 35.0, 0, days=1)
        bmi = 85.0 / 1.73**2
        assert 25.0 <= bmi < 30.0
        assert result["BMI_Category"][0][0] == "Pre-Obese"

    def test_obese(self):
        # BMI >= 30
        result = adult_weight(100.0, 1.73, 35.0, 0, days=1)
        bmi = 100.0 / 1.73**2
        assert bmi >= 30.0
        assert result["BMI_Category"][0][0] == "Obese"


class TestEIChange:
    """Test that non-zero EIchange causes weight change."""

    def test_positive_eichange_gains_weight(self):
        nsteps = 366  # ceil(365/1) + 1
        ei_change = np.full((nsteps, 1), 500.0)  # +500 kcal/day
        result = adult_weight(76.0, 1.73, 36.0, 0, EIchange=ei_change, days=365)
        assert result["Body_Weight"][-1, 0] > 76.0

    def test_negative_eichange_loses_weight(self):
        nsteps = 366
        ei_change = np.full((nsteps, 1), -500.0)  # -500 kcal/day
        result = adult_weight(76.0, 1.73, 36.0, 0, EIchange=ei_change, days=365)
        assert result["Body_Weight"][-1, 0] < 76.0
