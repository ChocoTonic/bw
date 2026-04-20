"""Physical constants for body weight dynamics models."""

# Adult model constants (converted from kJ to kcal)
roG = 4206.501       # Glycogen energy density (kcal/kg): 1000*17.6*0.23900573614
roF = 9440.727       # Fat energy density (kcal/kg): 1000*39.5*0.23900573614
roL = 1816.444       # Lean tissue energy density (kcal/kg): 1000*7.6*0.23900573614
gammaF = 3.107075    # Fat oxidation cost (kcal/kg/day): 13*0.23900573614
gammaL = 21.98853    # Lean tissue oxidation cost (kcal/kg/day): 92*0.23900573614
etaF = 179.2543      # Fat tissue synthesis cost (kcal/kg): 750*0.23900573614
etaL = 229.4455      # Lean tissue synthesis cost (kcal/kg): 960*0.23900573614
betaTEF = 0.1        # Thermal effect of feeding fraction
betaAT = 0.14        # Adaptive thermogenesis fraction
tauAT = 14.0         # Adaptive thermogenesis time constant (days)
Na_mg = 3220.0       # Sodium constant (mg): 1000*3.22
zetaNa = 3000.0      # ECF sodium sensitivity
zetaCI = 4000.0      # ECF carb intake sensitivity
G_BASELINE = 0.5     # Baseline glycogen (kg)

# RMR coefficients (Mifflin-St Jeor)
RMR_BW = 9.99
RMR_HT = 625.0
RMR_AGE = 4.92
RMR_MALE_INTERCEPT = 5.0
RMR_FEMALE_INTERCEPT = 161.0  # subtracted for females

# Forbes partition constant
C_FORBES = 10.4 * (roL / roF)

# Auxiliary constants
ALFA1 = -(1 + etaL / roL) * C_FORBES
ALFA2 = -(1 + etaF / roF)

# Child model constants
rhoFM = 9400.0       # Fat energy density for children (kcal/kg): 9.4*1000
DELTA_MIN = 10.0     # Minimum activity thermogenesis
P_CHILD = 12.0       # Sigmoid midpoint parameter
H_CHILD = 10.0       # Sigmoid steepness

# Reference FFM values at ages 2-18 (kg)
FFM_REF_MALE = [10.134, 12.099, 14.0, 16.0, 17.9, 19.9, 22.0, 24.4, 27.5, 29.5, 33.2, 38.1, 43.6, 49.1, 54.0, 57.7, 60.0]
FFM_REF_FEMALE = [9.477, 11.494, 13.2, 14.7, 16.3, 18.2, 20.5, 23.3, 26.4, 28.5, 32.4, 36.1, 38.9, 40.7, 41.7, 42.3, 42.6]

# Reference FM values at ages 2-18 (kg)
FM_REF_MALE = [2.456, 2.576, 2.7, 2.7, 2.8, 2.9, 3.3, 3.7, 4.8, 5.9, 6.7, 7.0, 7.2, 7.5, 8.0, 8.4, 8.8]
FM_REF_FEMALE = [2.433, 2.606, 2.8, 2.9, 3.2, 3.7, 4.3, 5.2, 7.2, 8.5, 9.2, 10.0, 11.3, 12.8, 14.0, 14.3, 14.3]

# Child growth parameters from "Dynamics" paper (sex-specific: [male, female])
GROWTH_A = {'male': 3.2, 'female': 2.3}
GROWTH_B = {'male': 9.6, 'female': 8.4}
GROWTH_D = {'male': 10.1, 'female': 1.1}
GROWTH_tA = {'male': 4.7, 'female': 4.5}
GROWTH_tB = {'male': 12.5, 'female': 11.7}
GROWTH_tD = {'male': 15.0, 'female': 16.2}
GROWTH_tauA = {'male': 2.5, 'female': 1.0}
GROWTH_tauB = {'male': 1.0, 'female': 0.9}
GROWTH_tauD = {'male': 1.5, 'female': 0.7}

# Child energy balance parameters from "Impact" paper
EB_A = {'male': 7.2, 'female': 16.5}
EB_B = {'male': 30.0, 'female': 47.0}
EB_D = {'male': 21.0, 'female': 41.0}
EB_tA = {'male': 5.6, 'female': 4.8}
EB_tB = {'male': 9.8, 'female': 9.1}
EB_tD = {'male': 15.0, 'female': 13.5}
EB_tauA = {'male': 15.0, 'female': 7.0}
EB_tauB = {'male': 1.5, 'female': 1.0}
EB_tauD = {'male': 2.0, 'female': 1.5}

# Child growth parameters from "Impact" paper (for actual integration)
GROWTH1_A = {'male': 3.2, 'female': 2.3}
GROWTH1_B = {'male': 9.6, 'female': 8.4}
GROWTH1_D = {'male': 10.0, 'female': 1.1}
GROWTH1_tA = {'male': 4.7, 'female': 4.5}
GROWTH1_tB = {'male': 12.5, 'female': 11.7}
GROWTH1_tD = {'male': 15.0, 'female': 16.0}
GROWTH1_tauA = {'male': 1.0, 'female': 1.0}
GROWTH1_tauB = {'male': 0.94, 'female': 0.94}
GROWTH1_tauD = {'male': 0.69, 'female': 0.69}

# Child sex-specific baseline constants
K_CHILD = {'male': 800.0, 'female': 700.0}
DELTA_MAX = {'male': 19.0, 'female': 17.0}

# Linear regression coefficients for FFM/FM reference (legacy, not used for interpolation)
FFM_BETA0 = {'male': 2.9, 'female': 3.8}
FFM_BETA1 = {'male': 2.9, 'female': 2.3}
FM_BETA0 = {'male': 1.2, 'female': 0.56}
FM_BETA1 = {'male': 0.41, 'female': 0.74}
