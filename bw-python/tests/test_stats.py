"""Tests for stats and plot modules."""

import numpy as np
import pandas as pd
import pytest

from bw.adult_model import adult_weight
from bw.child_model import child_weight
from bw.stats import model_mean, adult_bmi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_adult_model():
    """Single-individual adult model run for 30 days."""
    return adult_weight(80.0, 1.80, 40.0, "male",
                        EIchange=np.full((32, 1), -100.0),
                        days=30)


@pytest.fixture
def multi_adult_model():
    """Five-individual adult model run for 30 days."""
    bw = [45.0, 67.0, 58.0, 92.0, 81.0]
    ht = [1.30, 1.73, 1.77, 1.92, 1.73]
    age = [45.0, 23.0, 66.0, 44.0, 23.0]
    sex = ["male", "female", "female", "male", "male"]
    EIchange = np.column_stack([np.full(32, v) for v in [-100, -200, -200, -123, -50]])
    return adult_weight(bw, ht, age, sex, EIchange=EIchange, days=30)


@pytest.fixture
def child_model_fixture():
    """Two-child model run for 30 days."""
    return child_weight([5.0, 8.0], ["male", "female"], days=30)


# ---------------------------------------------------------------------------
# model_mean tests
# ---------------------------------------------------------------------------

class TestModelMean:

    def test_returns_dataframe_with_columns(self, single_adult_model):
        df = model_mean(single_adult_model)
        for col in ("variable", "day", "group", "mean", "SE_mean", "lower", "upper"):
            assert col in df.columns, f"Missing column: {col}"

    def test_default_vars_excludes_metadata(self, single_adult_model):
        df = model_mean(single_adult_model)
        assert "BMI_Category" not in df["variable"].values
        assert "Correct_Values" not in df["variable"].values
        assert "Time" not in df["variable"].values

    def test_single_individual_means_reasonable(self, single_adult_model):
        df = model_mean(single_adult_model, vars=["Body_Weight"])
        # Body weight should stay near 80 kg after only 30 days
        assert (df["mean"] > 50).all()
        assert (df["mean"] < 120).all()

    def test_specific_days(self, single_adult_model):
        df = model_mean(single_adult_model, vars=["Body_Weight"], days=[0, 10, 20, 30])
        assert set(df["day"].unique()).issubset(
            set(single_adult_model["Time"])
        )
        assert len(df) == 4  # 1 var * 4 days * 1 group

    def test_with_group(self, multi_adult_model):
        groups = np.array([0, 0, 1, 1, 1])
        df = model_mean(multi_adult_model, vars=["Body_Weight"],
                        group=groups, days=[0, 15, 30])
        # 1 var * 3 days * 2 groups = 6 rows
        assert len(df) == 6
        assert set(df["group"].unique()) == {0, 1}

    def test_with_weights(self, multi_adult_model):
        w = np.array([2.0, 1.0, 1.0, 3.0, 1.0])
        df_weighted = model_mean(multi_adult_model, vars=["Body_Weight"],
                                 weights=w, days=[0])
        df_equal = model_mean(multi_adult_model, vars=["Body_Weight"],
                              days=[0])
        # Weighted and unweighted means should generally differ
        # (unless all weights are equal or values identical)
        # Just check it runs and returns valid data
        assert len(df_weighted) == 1
        assert np.isfinite(df_weighted["mean"].iloc[0])

    def test_child_model(self, child_model_fixture):
        df = model_mean(child_model_fixture, vars=["Body_Weight"])
        assert len(df) > 0
        assert (df["mean"] > 0).all()

    def test_invalid_confidence(self, single_adult_model):
        with pytest.raises(ValueError):
            model_mean(single_adult_model, confidence=1.5)

    def test_bmi_category_rejected(self, single_adult_model):
        with pytest.raises(ValueError, match="BMI_Category"):
            model_mean(single_adult_model, vars=["BMI_Category"])


# ---------------------------------------------------------------------------
# adult_bmi tests
# ---------------------------------------------------------------------------

class TestAdultBMI:

    def test_returns_expected_columns(self, multi_adult_model):
        df = adult_bmi(multi_adult_model)
        for col in ("day", "group", "category", "proportion", "se", "lower", "upper"):
            assert col in df.columns

    def test_categories_are_valid(self, multi_adult_model):
        df = adult_bmi(multi_adult_model)
        valid = {"Underweight", "Normal", "Pre-Obese", "Obese"}
        assert set(df["category"].unique()).issubset(valid)

    def test_proportions_sum_to_one(self, multi_adult_model):
        df = adult_bmi(multi_adult_model, days=[0])
        total = df.groupby(["day", "group"])["proportion"].sum()
        np.testing.assert_allclose(total.values, 1.0, atol=1e-12)

    def test_with_group(self, multi_adult_model):
        groups = np.array([0, 0, 1, 1, 1])
        df = adult_bmi(multi_adult_model, group=groups, days=[0])
        assert set(df["group"].unique()) == {0, 1}
        # Each group-day should have 4 categories
        for g in [0, 1]:
            sub = df[df["group"] == g]
            assert len(sub) == 4

    def test_proportions_bounded(self, multi_adult_model):
        df = adult_bmi(multi_adult_model)
        assert (df["proportion"] >= 0).all()
        assert (df["proportion"] <= 1).all()
        assert (df["lower"] >= 0).all()
        assert (df["upper"] <= 1).all()


# ---------------------------------------------------------------------------
# model_plot tests
# ---------------------------------------------------------------------------

class TestModelPlot:

    def test_returns_figure(self, single_adult_model):
        import matplotlib
        matplotlib.use("Agg")
        from bw.plot import model_plot
        fig = model_plot(single_adult_model)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)

    def test_with_vars(self, multi_adult_model):
        import matplotlib
        matplotlib.use("Agg")
        from bw.plot import model_plot
        fig = model_plot(multi_adult_model, vars=["Body_Weight", "Fat_Mass"])
        axes = [ax for ax in fig.get_axes() if ax.get_visible()]
        assert len(axes) == 2
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)

    def test_child_model(self, child_model_fixture):
        import matplotlib
        matplotlib.use("Agg")
        from bw.plot import model_plot
        fig = model_plot(child_model_fixture)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)

    def test_single_var_ncol_1(self, single_adult_model):
        import matplotlib
        matplotlib.use("Agg")
        from bw.plot import model_plot
        fig = model_plot(single_adult_model, vars=["Body_Weight"], ncol=1)
        axes = [ax for ax in fig.get_axes() if ax.get_visible()]
        assert len(axes) == 1
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)
