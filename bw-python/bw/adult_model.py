"""Adult body weight change model (Hall et al.)."""

import math
import numpy as np

from bw.constants import (
    roG, roF, roL, gammaF, gammaL, etaF, etaL,
    betaTEF, betaAT, tauAT, Na_mg, zetaNa, zetaCI,
    G_BASELINE, RMR_BW, RMR_HT, RMR_AGE,
    RMR_MALE_INTERCEPT, RMR_FEMALE_INTERCEPT,
    C_FORBES, ALFA1, ALFA2,
)


def estimate_rmr(bw, ht, age, sex):
    """Estimate resting metabolic rate via Mifflin-St Jeor.

    Parameters
    ----------
    bw : array-like  Body weight (kg)
    ht : array-like  Height (m)
    age : array-like  Age (years)
    sex : array-like  0=male, 1=female

    Returns
    -------
    np.ndarray  RMR in kcal/day
    """
    bw, ht, age, sex = (np.asarray(x, dtype=float) for x in (bw, ht, age, sex))
    return ((RMR_BW * bw + RMR_HT * ht - RMR_AGE * age + RMR_MALE_INTERCEPT) * (1 - sex)
            + (RMR_BW * bw + RMR_HT * ht - RMR_AGE * age - RMR_FEMALE_INTERCEPT) * sex)


def estimate_fat(bw, ht, age, sex):
    """Estimate fat mass (kg).

    Parameters
    ----------
    bw : array-like  Body weight (kg)
    ht : array-like  Height (m)
    age : array-like  Age (years)
    sex : array-like  0=male, 1=female

    Returns
    -------
    np.ndarray  Fat mass in kg
    """
    bw, ht, age, sex = (np.asarray(x, dtype=float) for x in (bw, ht, age, sex))
    bmi = bw / (ht ** 2.0)
    male_fat = bw * (0.14 * age + 37.31 * np.log(bmi) - 103.94) / 100.0
    female_fat = bw * (0.14 * age + 39.96 * np.log(bmi) - 102.01) / 100.0
    return male_fat * (1 - sex) + female_fat * sex


def estimate_ecf(bw, ht, age, sex):
    """Estimate extracellular fluid (kg).

    Parameters
    ----------
    bw : array-like  Body weight (kg)
    ht : array-like  Height (m)
    age : array-like  Age (years)
    sex : array-like  0=male, 1=female

    Returns
    -------
    np.ndarray  ECF in kg
    """
    bw, ht, age, sex = (np.asarray(x, dtype=float) for x in (bw, ht, age, sex))
    male_ecf = 0.025 * age + 9.57 * ht + 0.191 * bw - 12.4
    female_ecf = -4.0 + 5.98 * ht + 0.167 * bw
    return male_ecf * (1.0 - sex) + female_ecf * sex


def _classify_bmi(bmi):
    """Classify BMI into categories.

    Parameters
    ----------
    bmi : np.ndarray, shape (nind,)

    Returns
    -------
    list of str
    """
    cats = []
    for val in np.atleast_1d(bmi):
        if val < 18.5:
            cats.append("Underweight")
        elif val < 25:
            cats.append("Normal")
        elif val < 30:
            cats.append("Pre-Obese")
        else:
            cats.append("Obese")
    return cats


class Adult:
    """Adult dynamic body weight model (Hall et al.).

    All arrays are 1-D with length nind (number of individuals).
    EIchange and NAchange are 2-D matrices with shape (nsteps, nind)
    where rows are time steps and columns are individuals.
    """

    def __init__(self, bw, ht, age, sex, EIchange, NAchange,
                 PAL, pcarb, pcarb_base, dt,
                 EI=None, fat=None, check_values=True):
        # Store inputs as float arrays
        self.bw = np.asarray(bw, dtype=float)
        self.ht = np.asarray(ht, dtype=float)
        self.age = np.asarray(age, dtype=float)
        self.sex = np.asarray(sex, dtype=float)
        self.EIchange = np.asarray(EIchange, dtype=float)
        self.NAchange = np.asarray(NAchange, dtype=float)
        self.PAL = np.asarray(PAL, dtype=float)
        self.pcarb = np.asarray(pcarb, dtype=float)
        self.pcarb_base = np.asarray(pcarb_base, dtype=float)
        self.dt = float(dt)
        self.check = check_values

        self.nind = self.bw.size
        self.G_base = np.full(self.nind, G_BASELINE)

        # Derived quantities
        self.rmr = estimate_rmr(self.bw, self.ht, self.age, self.sex)
        self.atinit = np.zeros(self.nind)
        self.ecfinit = estimate_ecf(self.bw, self.ht, self.age, self.sex)

        have_EI = EI is not None
        have_fat = fat is not None

        if have_EI and have_fat:
            self.EI = np.asarray(EI, dtype=float)
            self.fat = np.asarray(fat, dtype=float)
            self.lean = self.bw - (self.ecfinit + self.fat + 3.7 * self.G_base)
        elif have_EI and not have_fat:
            self.EI = np.asarray(EI, dtype=float)
            self.fat = estimate_fat(self.bw, self.ht, self.age, self.sex)
            self.lean = self.bw - (self.ecfinit + self.fat + 3.7 * self.G_base)
        elif have_fat and not have_EI:
            self.steady_state = self.rmr * self.PAL
            self.EI = self.steady_state.copy()
            self.fat = np.asarray(fat, dtype=float)
            self.lean = self.bw - (self.ecfinit + self.fat + 3.7 * self.G_base)
        else:
            self.steady_state = self.rmr * self.PAL
            self.EI = self.steady_state.copy()
            self.fat = estimate_fat(self.bw, self.ht, self.age, self.sex)
            self.lean = self.bw - (self.ecfinit + self.fat + 3.7 * self.G_base)

        self.delta = ((1.0 - betaTEF) * self.PAL - 1.0) * self.rmr / self.bw
        self.K = self.rmr * self.PAL - gammaL * self.lean - gammaF * self.fat - self.delta * self.bw
        self.CIb = self.pcarb_base * self.EI
        self.kG = self.CIb / (self.G_base ** 2.0)

    # ---- helper functions matching C++ exactly ----

    def _deltaEI(self, t):
        idx = int(math.floor(t / self.dt))
        return self.EIchange[idx, :]

    def _deltaNA(self, t):
        idx = int(math.floor(t / self.dt))
        return self.NAchange[idx, :]

    def _TotalIntake(self, t):
        return self.EI + self._deltaEI(t)

    def _CI(self, t):
        return self.pcarb * self._TotalIntake(t)

    def _TEF(self, t):
        return betaTEF * self._deltaEI(t)

    def _fatMass(self, L):
        return self.fat * np.exp(roL * (L - self.lean) / (roF * C_FORBES))

    def _dG(self, t, G):
        return (self._CI(t) - self.kG * G ** 2.0) / roG

    def _dAT(self, t, AT):
        return (betaAT * self._deltaEI(t) - AT) * (1.0 / tauAT)

    def _dECF(self, t, ECF):
        return (self._deltaNA(t) - zetaNa * (ECF - self.ecfinit)
                - zetaCI * (1.0 - self._CI(t) / self.CIb)) / Na_mg

    def _R(self, t, L, G, AT, ECF):
        F = self._fatMass(L)
        weight = L + F + ECF + 3.7 * G
        R3 = (self.K + self.delta * weight + self._TEF(t) + AT
              - self._TotalIntake(t) + roG * self._dG(t, G))
        return (R3 + gammaL * L + gammaF * F) / (ALFA1 + ALFA2 * F)

    def _dL(self, t, L, G, AT, ECF):
        return self._R(t, L, G, AT, ECF) * (C_FORBES / roL)

    # ---- RK4 solver matching C++ exactly ----

    def rk4(self, days):
        """Run the RK4 solver for the given number of days.

        Returns
        -------
        dict with simulation results
        """
        dt = self.dt
        nsims = min(int(math.ceil(days / dt)),
                    self.EIchange.shape[0] - 1)
        nind = self.nind
        nsteps = nsims + 1

        # Allocate output arrays: shape (nsteps, nind)
        AT_arr = np.zeros((nsteps, nind))
        ECF_arr = np.zeros((nsteps, nind))
        GLY_arr = np.zeros((nsteps, nind))
        L_arr = np.zeros((nsteps, nind))
        F_arr = np.zeros((nsteps, nind))
        BW_arr = np.zeros((nsteps, nind))
        BMI_arr = np.zeros((nsteps, nind))
        TEI_arr = np.zeros((nsteps, nind))
        AGE_arr = np.zeros((nsteps, nind))
        CAT_arr = [None] * nsteps
        TIME_arr = np.zeros(nsteps)

        # Initial conditions
        AT_arr[0, :] = self.atinit
        ECF_arr[0, :] = self.ecfinit
        GLY_arr[0, :] = self.G_base
        L_arr[0, :] = self.lean
        F_arr[0, :] = self._fatMass(self.lean)
        BW_arr[0, :] = self.bw
        BMI_arr[0, :] = self.bw / (self.ht ** 2.0)
        CAT_arr[0] = _classify_bmi(BMI_arr[0, :])
        TEI_arr[0, :] = self.EI
        AGE_arr[0, :] = self.age
        TIME_arr[0] = 0.0

        correct_vals = True

        for i in range(1, nsims + 1):
            t_prev = TIME_arr[i - 1]

            # --- Adaptive Thermogenesis (independent RK4) ---
            at_prev = AT_arr[i - 1, :]
            k1 = self._dAT(t_prev, at_prev)
            k2 = self._dAT(t_prev + 0.5 * dt, at_prev + 0.5 * dt * k1)
            k3 = self._dAT(t_prev + 0.5 * dt, at_prev + 0.5 * dt * k2)
            k4 = self._dAT(t_prev + dt, at_prev + dt * k3)
            AT_arr[i, :] = at_prev + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            # --- ECF (independent RK4) ---
            ecf_prev = ECF_arr[i - 1, :]
            k1 = self._dECF(t_prev, ecf_prev)
            k2 = self._dECF(t_prev + 0.5 * dt, ecf_prev + 0.5 * dt * k1)
            k3 = self._dECF(t_prev + 0.5 * dt, ecf_prev + 0.5 * dt * k2)
            k4 = self._dECF(t_prev + dt, ecf_prev + dt * k3)
            ECF_arr[i, :] = ecf_prev + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            # --- Glycogen (independent RK4) ---
            gly_prev = GLY_arr[i - 1, :]
            k1 = self._dG(t_prev, gly_prev)
            k2 = self._dG(t_prev + 0.5 * dt, gly_prev + 0.5 * dt * k1)
            k3 = self._dG(t_prev + 0.5 * dt, gly_prev + 0.5 * dt * k2)
            k4 = self._dG(t_prev + dt, gly_prev + dt * k3)
            GLY_arr[i, :] = gly_prev + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            # --- Lean mass (uses already-updated AT, ECF, GLY) ---
            l_prev = L_arr[i - 1, :]
            at_old = AT_arr[i - 1, :]
            at_new = AT_arr[i, :]
            ecf_old = ECF_arr[i - 1, :]
            ecf_new = ECF_arr[i, :]
            gly_old = GLY_arr[i - 1, :]
            gly_new = GLY_arr[i, :]

            k1 = self._dL(t_prev, l_prev, gly_old, at_old, ecf_old)
            k2 = self._dL(t_prev + 0.5 * dt, l_prev + 0.5 * dt * k1,
                          0.5 * (gly_new + gly_old),
                          0.5 * (at_new + at_old),
                          0.5 * (ecf_new + ecf_old))
            k3 = self._dL(t_prev + 0.5 * dt, l_prev + 0.5 * dt * k2,
                          0.5 * (gly_new + gly_old),
                          0.5 * (at_new + at_old),
                          0.5 * (ecf_new + ecf_old))
            k4 = self._dL(t_prev + dt, l_prev + dt * k3,
                          gly_new, at_new, ecf_new)
            L_arr[i, :] = l_prev + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            # Update derived quantities
            F_arr[i, :] = self._fatMass(L_arr[i, :])
            BW_arr[i, :] = F_arr[i, :] + L_arr[i, :] + ECF_arr[i, :] + 3.7 * GLY_arr[i, :]
            BMI_arr[i, :] = BW_arr[i, :] / (self.ht ** 2.0)
            CAT_arr[i] = _classify_bmi(BMI_arr[i, :])
            TIME_arr[i] = TIME_arr[i - 1] + dt
            AGE_arr[i, :] = AGE_arr[i - 1, :] + dt / 365.0
            TEI_arr[i, :] = self._TotalIntake(TIME_arr[i])

        return {
            "Time": TIME_arr,
            "Age": AGE_arr,
            "Body_Weight": BW_arr,
            "Lean_Mass": L_arr,
            "Fat_Mass": F_arr,
            "Glycogen": GLY_arr,
            "Extracellular_Fluid": ECF_arr,
            "Adaptive_Thermogenesis": AT_arr,
            "BMI": BMI_arr,
            "BMI_Category": CAT_arr,
            "Energy_Intake": TEI_arr,
            "Correct_Values": correct_vals,
            "Model_Type": "Adult",
        }


def adult_weight(bw, ht, age, sex, *, EIchange=None, NAchange=None,
                 EI=None, fat=None, PAL=1.5, pcarb_base=0.5, pcarb=None,
                 days=365, dt=1.0, check_values=True):
    """Run the adult body weight change model.

    Parameters
    ----------
    bw : float or array-like  Body weight (kg)
    ht : float or array-like  Height (m)
    age : float or array-like  Age (years)
    sex : str, int, or array-like  "male"/"female" or 0/1
    EIchange : array-like, optional  Energy intake change matrix (nsteps, nind)
    NAchange : array-like, optional  Sodium change matrix (nsteps, nind)
    EI : float or array-like, optional  Baseline energy intake (kcal)
    fat : float or array-like, optional  Baseline fat mass (kg)
    PAL : float or array-like  Physical activity level (default 1.5)
    pcarb_base : float or array-like  Baseline carbohydrate fraction (default 0.5)
    pcarb : float or array-like, optional  Carbohydrate fraction during simulation
    days : int  Number of days to simulate (default 365)
    dt : float  Time step in days (default 1.0)
    check_values : bool  Whether to check for invalid values (default True)

    Returns
    -------
    dict  Simulation results
    """
    # Convert sex strings to numeric
    def _convert_sex(s):
        if isinstance(s, str):
            s = s.lower()
            if s == "male":
                return 0.0
            elif s == "female":
                return 1.0
            else:
                raise ValueError(f"Unknown sex string: '{s}'. Use 'male' or 'female'.")
        return float(s)

    # Handle sex conversion for arrays/lists of strings
    if isinstance(sex, (list, tuple)):
        sex = np.array([_convert_sex(s) for s in sex], dtype=float)
    elif isinstance(sex, str):
        sex = np.array([_convert_sex(sex)], dtype=float)
    elif isinstance(sex, np.ndarray) and sex.dtype.kind in ('U', 'S', 'O'):
        sex = np.array([_convert_sex(s) for s in sex.flat], dtype=float)
    else:
        sex = np.atleast_1d(np.asarray(sex, dtype=float))

    # Convert scalars to 1-D arrays
    bw = np.atleast_1d(np.asarray(bw, dtype=float))
    ht = np.atleast_1d(np.asarray(ht, dtype=float))
    age = np.atleast_1d(np.asarray(age, dtype=float))
    PAL = np.atleast_1d(np.asarray(PAL, dtype=float))
    pcarb_base = np.atleast_1d(np.asarray(pcarb_base, dtype=float))

    nind = bw.size

    # Broadcast scalar PAL / pcarb_base
    if PAL.size == 1 and nind > 1:
        PAL = np.full(nind, PAL[0])
    if pcarb_base.size == 1 and nind > 1:
        pcarb_base = np.full(nind, pcarb_base[0])

    # pcarb defaults to pcarb_base
    if pcarb is None:
        pcarb = pcarb_base.copy()
    else:
        pcarb = np.atleast_1d(np.asarray(pcarb, dtype=float))
        if pcarb.size == 1 and nind > 1:
            pcarb = np.full(nind, pcarb[0])

    # Default EIchange / NAchange
    # Match R: ncol = ceiling(days/dt), transposed to nrow for C++
    nsteps_ei = int(math.ceil(days / dt))
    if EIchange is None:
        EIchange = np.zeros((nsteps_ei, nind))
    else:
        EIchange = np.asarray(EIchange, dtype=float)
    if NAchange is None:
        # Match EIchange dimensions (R requires same dimensions)
        NAchange = np.zeros_like(EIchange)
    else:
        NAchange = np.asarray(NAchange, dtype=float)

    # Validate
    if bw.size != ht.size or bw.size != age.size or bw.size != sex.size:
        raise ValueError("bw, ht, age, and sex must have the same length.")
    if np.any(bw <= 0):
        raise ValueError("Body weight must be positive.")
    if np.any(ht <= 0):
        raise ValueError("Height must be positive.")
    if np.any(age < 0):
        raise ValueError("Age must be non-negative.")
    if np.any((sex != 0) & (sex != 1)):
        raise ValueError("sex must be 0 (male) or 1 (female).")
    if np.any(PAL <= 0):
        raise ValueError("PAL must be positive.")

    # Handle optional EI / fat
    ei_arg = None
    fat_arg = None
    if EI is not None:
        ei_arg = np.atleast_1d(np.asarray(EI, dtype=float))
        if ei_arg.size == 1 and nind > 1:
            ei_arg = np.full(nind, ei_arg[0])
    if fat is not None:
        fat_arg = np.atleast_1d(np.asarray(fat, dtype=float))
        if fat_arg.size == 1 and nind > 1:
            fat_arg = np.full(nind, fat_arg[0])

    model = Adult(bw, ht, age, sex, EIchange, NAchange,
                  PAL, pcarb, pcarb_base, dt,
                  EI=ei_arg, fat=fat_arg, check_values=check_values)

    return model.rk4(days)
