"""Child weight dynamics model.

Implements the Hall et al. (2013) dynamic model for childhood growth
and obesity, solved via 4th-order Runge-Kutta.
"""
import math
import numpy as np

from bw.constants import (
    rhoFM,
    DELTA_MIN,
    P_CHILD,
    H_CHILD,
    FFM_REF_MALE,
    FFM_REF_FEMALE,
    FM_REF_MALE,
    FM_REF_FEMALE,
    GROWTH_A, GROWTH_B, GROWTH_D,
    GROWTH_tA, GROWTH_tB, GROWTH_tD,
    GROWTH_tauA, GROWTH_tauB, GROWTH_tauD,
    EB_A, EB_B, EB_D,
    EB_tA, EB_tB, EB_tD,
    EB_tauA, EB_tauB, EB_tauD,
    K_CHILD,
    DELTA_MAX,
)


# ---------------------------------------------------------------------------
# Reference curve helpers
# ---------------------------------------------------------------------------

def child_reference_FFM_FM(age, sex):
    """Compute reference FFM and FM for given ages and sexes.

    Parameters
    ----------
    age : array-like
        Ages in years (typically 2-18).
    sex : array-like
        0 = male, 1 = female.

    Returns
    -------
    dict with 'FFM' and 'FM' arrays of shape (n,).
    """
    age = np.atleast_1d(np.asarray(age, dtype=float))
    sex = np.atleast_1d(np.asarray(sex, dtype=float))
    n = len(age)

    ffm_ref_male = np.array(FFM_REF_MALE)
    ffm_ref_female = np.array(FFM_REF_FEMALE)
    fm_ref_male = np.array(FM_REF_MALE)
    fm_ref_female = np.array(FM_REF_FEMALE)

    ffm = np.zeros(n)
    fm = np.zeros(n)

    for i in range(n):
        s = sex[i]
        ref_ffm = ffm_ref_male * (1 - s) + ffm_ref_female * s
        ref_fm = fm_ref_male * (1 - s) + fm_ref_female * s

        ffm[i] = _interp_ref(age[i], ref_ffm)
        fm[i] = _interp_ref(age[i], ref_fm)

    return {"FFM": ffm, "FM": fm}


def _interp_ref(t, ref):
    """Piecewise linear interpolation matching C++ FFMReference/FMReference."""
    if t >= 18.0:
        return ref[16]
    jmin = int(math.floor(t))
    jmin = max(jmin, 2)
    jmin = jmin - 2
    jmax = min(jmin + 1, 17)
    diff = t - math.floor(t)
    return ref[jmin] + diff * (ref[jmax] - ref[jmin])


def child_reference_EI(age, sex, FM, FFM, days, dt=1.0):
    """Compute reference energy intake matrix.

    Returns
    -------
    np.ndarray of shape (n_individuals, ceil(days/dt) + 1)
    """
    age = np.atleast_1d(np.asarray(age, dtype=float))
    sex = np.atleast_1d(np.asarray(sex, dtype=float))
    FM = np.atleast_1d(np.asarray(FM, dtype=float))
    FFM = np.atleast_1d(np.asarray(FFM, dtype=float))

    nind = len(age)
    nsims = int(math.floor(days / dt))
    EI = np.zeros((nsims + 1, nind))

    # Build a temporary Child with dummy EIntake just to call IntakeReference
    # Actually, IntakeReference doesn't depend on EIntake, so we build
    # a helper that computes it directly.
    child = Child(age, sex, FFM, FM, EI, dt, check_values=False)

    for i in range(nsims + 1):
        t = age + i * dt / 365.0
        EI[i, :] = child._intake_reference(t)

    return EI


# ---------------------------------------------------------------------------
# Core math functions (module-level, vectorized over individuals)
# ---------------------------------------------------------------------------

def _general_ode(t, A, B, D, tA, tB, tD, tauA, tauB, tauD):
    """General ODE term: sum of exponential + two Gaussians."""
    return (A * np.exp(-(t - tA) / tauA)
            + B * np.exp(-0.5 * ((t - tB) / tauB) ** 2)
            + D * np.exp(-0.5 * ((t - tD) / tauD) ** 2))


def _sex_param(male_val, female_val, sex):
    """Compute sex-specific parameter: male*(1-sex) + female*sex."""
    return male_val * (1.0 - sex) + female_val * sex


# ---------------------------------------------------------------------------
# Child class
# ---------------------------------------------------------------------------

class Child:
    """Child weight dynamics model.

    Parameters
    ----------
    age : array-like
        Initial ages in years.
    sex : array-like
        0 = male, 1 = female.
    FFM : array-like
        Initial fat-free mass (kg).
    FM : array-like
        Initial fat mass (kg).
    EIntake : np.ndarray or None
        Energy intake matrix, shape (n_timesteps, n_individuals).
    dt : float
        Time step in days.
    check_values : bool
        Whether to check for invalid values during integration.
    K_logistic, Q_logistic, A_logistic, B_logistic, nu_logistic, C_logistic : float or None
        Richardson generalized logistic curve parameters. If all provided,
        use logistic intake instead of EIntake matrix.
    """

    def __init__(self, age, sex, FFM, FM, EIntake=None, dt=1.0,
                 check_values=True, *,
                 K_logistic=None, Q_logistic=None, A_logistic=None,
                 B_logistic=None, nu_logistic=None, C_logistic=None):
        self.age = np.atleast_1d(np.asarray(age, dtype=float))
        self.sex = np.atleast_1d(np.asarray(sex, dtype=float))
        self.FFM = np.atleast_1d(np.asarray(FFM, dtype=float))
        self.FM = np.atleast_1d(np.asarray(FM, dtype=float))
        self.dt = float(dt)
        self.check = check_values
        self.nind = len(self.age)

        # Determine intake mode
        logistic_params = [K_logistic, Q_logistic, A_logistic,
                           B_logistic, nu_logistic, C_logistic]
        if all(p is not None for p in logistic_params):
            self.generalized_logistic = True
            self.K_logistic = float(K_logistic)
            self.Q_logistic = float(Q_logistic)
            self.A_logistic = float(A_logistic)
            self.B_logistic = float(B_logistic)
            self.nu_logistic = float(nu_logistic)
            self.C_logistic = float(C_logistic)
            self.EIntake = None
        else:
            self.generalized_logistic = False
            if EIntake is not None:
                self.EIntake = np.asarray(EIntake, dtype=float)
            else:
                self.EIntake = None

        self._build()

    def _build(self):
        """Initialize sex-specific parameters."""
        sex = self.sex

        self.K = _sex_param(K_CHILD['male'], K_CHILD['female'], sex)
        self.deltamax = _sex_param(DELTA_MAX['male'], DELTA_MAX['female'], sex)

        # Growth dynamic parameters
        self._A = _sex_param(GROWTH_A['male'], GROWTH_A['female'], sex)
        self._B = _sex_param(GROWTH_B['male'], GROWTH_B['female'], sex)
        self._D = _sex_param(GROWTH_D['male'], GROWTH_D['female'], sex)
        self._tA = _sex_param(GROWTH_tA['male'], GROWTH_tA['female'], sex)
        self._tB = _sex_param(GROWTH_tB['male'], GROWTH_tB['female'], sex)
        self._tD = _sex_param(GROWTH_tD['male'], GROWTH_tD['female'], sex)
        self._tauA = _sex_param(GROWTH_tauA['male'], GROWTH_tauA['female'], sex)
        self._tauB = _sex_param(GROWTH_tauB['male'], GROWTH_tauB['female'], sex)
        self._tauD = _sex_param(GROWTH_tauD['male'], GROWTH_tauD['female'], sex)

        # EB impact parameters
        self._A_EB = _sex_param(EB_A['male'], EB_A['female'], sex)
        self._B_EB = _sex_param(EB_B['male'], EB_B['female'], sex)
        self._D_EB = _sex_param(EB_D['male'], EB_D['female'], sex)
        self._tA_EB = _sex_param(EB_tA['male'], EB_tA['female'], sex)
        self._tB_EB = _sex_param(EB_tB['male'], EB_tB['female'], sex)
        self._tD_EB = _sex_param(EB_tD['male'], EB_tD['female'], sex)
        self._tauA_EB = _sex_param(EB_tauA['male'], EB_tauA['female'], sex)
        self._tauB_EB = _sex_param(EB_tauB['male'], EB_tauB['female'], sex)
        self._tauD_EB = _sex_param(EB_tauD['male'], EB_tauD['female'], sex)

        # Pre-build reference tables per individual
        ffm_ref_m = np.array(FFM_REF_MALE)
        ffm_ref_f = np.array(FFM_REF_FEMALE)
        fm_ref_m = np.array(FM_REF_MALE)
        fm_ref_f = np.array(FM_REF_FEMALE)

        # Shape: (17, nind)
        self._ffm_ref = np.outer(ffm_ref_m, 1.0 - sex) + np.outer(ffm_ref_f, sex)
        self._fm_ref = np.outer(fm_ref_m, 1.0 - sex) + np.outer(fm_ref_f, sex)

    # -------------------------------------------------------------------
    # Internal math
    # -------------------------------------------------------------------

    def _growth_dynamic(self, t):
        return _general_ode(t, self._A, self._B, self._D,
                            self._tA, self._tB, self._tD,
                            self._tauA, self._tauB, self._tauD)

    def _eb_impact(self, t):
        return _general_ode(t, self._A_EB, self._B_EB, self._D_EB,
                            self._tA_EB, self._tB_EB, self._tD_EB,
                            self._tauA_EB, self._tauB_EB, self._tauD_EB)

    @staticmethod
    def _cRhoFFM(FFM):
        return 4.3 * FFM + 837.0

    def _cP(self, FFM, FM):
        rhoFFM_val = self._cRhoFFM(FFM)
        C = 10.4 * rhoFFM_val / rhoFM
        return C / (C + FM)

    def _delta(self, t):
        return DELTA_MIN + (self.deltamax - DELTA_MIN) / (1.0 + (t / P_CHILD) ** H_CHILD)

    def _ffm_reference(self, t):
        """Piecewise linear interpolation of FFM reference table."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.zeros(self.nind)
        for i in range(self.nind):
            if t[i] >= 18.0:
                result[i] = self._ffm_ref[16, i]
            else:
                jmin = int(math.floor(t[i]))
                jmin = max(jmin, 2) - 2
                jmax = min(jmin + 1, 17)
                diff = t[i] - math.floor(t[i])
                result[i] = (self._ffm_ref[jmin, i]
                             + diff * (self._ffm_ref[jmax, i] - self._ffm_ref[jmin, i]))
        return result

    def _fm_reference(self, t):
        """Piecewise linear interpolation of FM reference table."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.zeros(self.nind)
        for i in range(self.nind):
            if t[i] >= 18.0:
                result[i] = self._fm_ref[16, i]
            else:
                jmin = int(math.floor(t[i]))
                jmin = max(jmin, 2) - 2
                jmax = min(jmin + 1, 17)
                diff = t[i] - math.floor(t[i])
                result[i] = (self._fm_ref[jmin, i]
                             + diff * (self._fm_ref[jmax, i] - self._fm_ref[jmin, i]))
        return result

    def _intake_reference(self, t):
        EB = self._eb_impact(t)
        FFMref = self._ffm_reference(t)
        FMref = self._fm_reference(t)
        delta = self._delta(t)
        growth = self._growth_dynamic(t)
        p = self._cP(FFMref, FMref)
        rhoFFM_val = self._cRhoFFM(FFMref)
        return (EB + self.K + (22.4 + delta) * FFMref + (4.5 + delta) * FMref
                + 230.0 / rhoFFM_val * (p * EB + growth)
                + 180.0 / rhoFM * ((1.0 - p) * EB - growth))

    def _intake(self, t):
        if self.generalized_logistic:
            return (self.A_logistic
                    + (self.K_logistic - self.A_logistic)
                    / (self.C_logistic + self.Q_logistic * np.exp(-self.B_logistic * t))
                    ** (1.0 / self.nu_logistic))
        else:
            timeval = int(math.floor(365.0 * (t[0] - self.age[0]) / self.dt))
            return self.EIntake[timeval, :]

    def _expenditure(self, t, FFM, FM):
        delta = self._delta(t)
        Iref = self._intake_reference(t)
        Intakeval = self._intake(t)
        DeltaI = Intakeval - Iref
        p = self._cP(FFM, FM)
        rhoFFM_val = self._cRhoFFM(FFM)
        growth = self._growth_dynamic(t)
        Expend = (self.K + (22.4 + delta) * FFM + (4.5 + delta) * FM
                  + 0.24 * DeltaI
                  + (230.0 / rhoFFM_val * p + 180.0 / rhoFM * (1.0 - p)) * Intakeval
                  + growth * (230.0 / rhoFFM_val - 180.0 / rhoFM))
        return Expend / (1.0 + 230.0 / rhoFFM_val * p + 180.0 / rhoFM * (1.0 - p))

    def _dMass(self, t, FFM, FM):
        """Return (dFFM, dFM) as two arrays of shape (nind,)."""
        rhoFFM_val = self._cRhoFFM(FFM)
        p = self._cP(FFM, FM)
        growth = self._growth_dynamic(t)
        expend = self._expenditure(t, FFM, FM)
        intake = self._intake(t)
        dFFM = (p * (intake - expend) + growth) / rhoFFM_val
        dFM = ((1.0 - p) * (intake - expend) - growth) / rhoFM
        return dFFM, dFM

    # -------------------------------------------------------------------
    # RK4 solver
    # -------------------------------------------------------------------

    def rk4(self, days):
        """Run RK4 integration.

        Returns
        -------
        dict with keys: Time, Age, Fat_Free_Mass, Fat_Mass, Body_Weight,
                        Correct_Values, Model_Type
        """
        # R wrapper calls rk4(days-1) to account for C++/R indexing difference
        nsims = int(math.floor((days - 1) / self.dt))
        dt = self.dt

        # Storage: (nind, nsims+1)
        ModelFFM = np.zeros((self.nind, nsims + 1))
        ModelFM = np.zeros((self.nind, nsims + 1))
        ModelBW = np.zeros((self.nind, nsims + 1))
        AGE = np.zeros((self.nind, nsims + 1))
        TIME = np.zeros(nsims + 1)

        # Initial conditions
        ModelFFM[:, 0] = self.FFM
        ModelFM[:, 0] = self.FM
        ModelBW[:, 0] = self.FFM + self.FM
        AGE[:, 0] = self.age
        TIME[0] = 0.0

        correct_vals = True

        for i in range(1, nsims + 1):
            age_prev = AGE[:, i - 1]
            ffm_prev = ModelFFM[:, i - 1]
            fm_prev = ModelFM[:, i - 1]

            k1_ffm, k1_fm = self._dMass(age_prev, ffm_prev, fm_prev)
            k2_ffm, k2_fm = self._dMass(age_prev + 0.5 * dt / 365.0,
                                          ffm_prev + 0.5 * k1_ffm,
                                          fm_prev + 0.5 * k1_fm)
            k3_ffm, k3_fm = self._dMass(age_prev + 0.5 * dt / 365.0,
                                          ffm_prev + 0.5 * k2_ffm,
                                          fm_prev + 0.5 * k2_fm)
            k4_ffm, k4_fm = self._dMass(age_prev + dt / 365.0,
                                          ffm_prev + k3_ffm,
                                          fm_prev + k3_fm)

            ModelFFM[:, i] = ffm_prev + dt * (k1_ffm + 2.0 * k2_ffm + 2.0 * k3_ffm + k4_ffm) / 6.0
            ModelFM[:, i] = fm_prev + dt * (k1_fm + 2.0 * k2_fm + 2.0 * k3_fm + k4_fm) / 6.0
            ModelBW[:, i] = ModelFFM[:, i] + ModelFM[:, i]

            TIME[i] = TIME[i - 1] + dt
            AGE[:, i] = AGE[:, i - 1] + dt / 365.0

        return {
            "Time": TIME,
            "Age": AGE,
            "Fat_Free_Mass": ModelFFM,
            "Fat_Mass": ModelFM,
            "Body_Weight": ModelBW,
            "Correct_Values": correct_vals,
            "Model_Type": "Children",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def child_weight(age, sex, *, FM=None, FFM=None, EI=None,
                 richardson_params=None, days=365, dt=1.0,
                 check_values=True):
    """Simulate child weight dynamics.

    Parameters
    ----------
    age : float or array-like
        Initial age(s) in years.
    sex : str or array-like of str
        "male" or "female" (or 0/1).
    FM : array-like or None
        Initial fat mass (kg). If None, use reference.
    FFM : array-like or None
        Initial fat-free mass (kg). If None, use reference.
    EI : np.ndarray or None
        Energy intake matrix. If None and richardson_params is None,
        compute reference intake.
    richardson_params : dict or None
        Keys: K, Q, A, B, nu, C for Richardson generalized logistic curve.
    days : int
        Number of days to simulate.
    dt : float
        Time step in days.
    check_values : bool
        Whether to validate intermediate results.

    Returns
    -------
    dict with keys: Time, Age, Fat_Free_Mass, Fat_Mass, Body_Weight,
                    Correct_Values, Model_Type
    """
    # Convert age to array
    age_arr = np.atleast_1d(np.asarray(age, dtype=float))

    # Convert sex to numeric
    sex_arr = _convert_sex(sex, len(age_arr))

    # Validate
    if len(age_arr) != len(sex_arr):
        raise ValueError("age and sex must have the same length")
    if np.any(age_arr < 2) or np.any(age_arr > 18):
        raise ValueError("age must be between 2 and 18 years")

    # Reference FFM/FM if needed
    if FM is None or FFM is None:
        ref = child_reference_FFM_FM(age_arr, sex_arr)
        if FFM is None:
            FFM = ref["FFM"]
        if FM is None:
            FM = ref["FM"]

    FM_arr = np.atleast_1d(np.asarray(FM, dtype=float))
    FFM_arr = np.atleast_1d(np.asarray(FFM, dtype=float))

    if len(FM_arr) != len(age_arr):
        raise ValueError("FM must have the same length as age")
    if len(FFM_arr) != len(age_arr):
        raise ValueError("FFM must have the same length as age")

    # Build Child object
    if richardson_params is not None:
        rp = richardson_params
        child = Child(age_arr, sex_arr, FFM_arr, FM_arr, dt=dt,
                      check_values=check_values,
                      K_logistic=rp['K'], Q_logistic=rp['Q'],
                      A_logistic=rp['A'], B_logistic=rp['B'],
                      nu_logistic=rp['nu'], C_logistic=rp['C'])
    else:
        if EI is None:
            EI = child_reference_EI(age_arr, sex_arr, FM_arr, FFM_arr, days, dt)
        child = Child(age_arr, sex_arr, FFM_arr, FM_arr, EI, dt=dt,
                      check_values=check_values)

    return child.rk4(days)


def _convert_sex(sex, n):
    """Convert sex to numeric array (0=male, 1=female)."""
    if isinstance(sex, str):
        sex = [sex] * n
    sex = np.atleast_1d(sex)
    result = np.zeros(len(sex), dtype=float)
    for i, s in enumerate(sex):
        if isinstance(s, str):
            sl = s.lower()
            if sl == "male":
                result[i] = 0.0
            elif sl == "female":
                result[i] = 1.0
            else:
                raise ValueError(f"sex must be 'male' or 'female', got '{s}'")
        else:
            val = float(s)
            if val not in (0.0, 1.0):
                raise ValueError(f"numeric sex must be 0 or 1, got {val}")
            result[i] = val
    return result
