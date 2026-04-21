# Formula Comparison: Hall et al. 2011 Lancet Web Appendix vs `bw` R Package

Comparison of the mathematical model in the supplementary web appendix to Hall KD, Sacks G, Chandramohan D, et al. "Quantification of the effect of energy imbalance on bodyweight" (Lancet 2011; 378: 826–37) against the C++ implementation in `src/adult_weight.cpp`.

## Equation-by-equation comparison

### Eq 1 — Glycogen dynamics

**Paper:**
$$\rho_G \frac{dG}{dt} = CI - k_G G^2$$

**Code** (`adult_weight.cpp:316`):
```cpp
return (CI(t) - kG*pow(G, 2.0))/roG;
```

**Verdict:** Match.

---

### Eq 2 — Extracellular fluid

**Paper:**
$$\frac{dECF}{dt} = \frac{1}{[Na]} \left( \Delta Na_{diet} - \zeta_{Na}(ECF - ECF_{init}) - \zeta_{CI}(1 - CI/CI_b) \right)$$

**Code** (`adult_weight.cpp:326`):
```cpp
return (deltaNA(t) - zetaNa*(ECF - ecfinit) - zetaCI*(1.0 - CI(t)/CIb)) / Na;
```

**Verdict:** Match.

---

### Eq 3 — Energy partitioning between fat and lean tissue

**Paper:**
$$\rho_F \frac{dF}{dt} = (1 - p)\left(EI - EE - \rho_G \frac{dG}{dt}\right)$$
$$\rho_L \frac{dL}{dt} = p\left(EI - EE - \rho_G \frac{dG}{dt}\right)$$

where $p = C/(C + F)$ with $C = 10.4 \cdot \rho_L / \rho_F$.

**Code** (`adult_weight.cpp:248, 360`):
```cpp
C = 10.4*(roL/roF);
// Integrated Forbes relation:
return fat * exp(roL * (L - lean)/(roF * C));
```

The code uses the integrated form of the Forbes partitioning equation rather than solving two coupled ODEs directly. Dividing the two equations in Eq 3 and integrating yields $F = F_0 \exp\left(\frac{\rho_L (L - L_0)}{\rho_F C}\right)$, which is what the code implements.

**Verdict:** Match (equivalent formulation).

---

### Eq 4 — Initial fat mass estimation (Jackson et al.)

**Paper:**
$$F_m = \frac{BW}{100}\left[0.14 \times age + 37.31 \times \ln(BW/H^2) - 103.94\right]$$
$$F_w = \frac{BW}{100}\left[0.14 \times age + 39.96 \times \ln(BW/H^2) - 102.01\right]$$

**Code** (`adult_weight.cpp:299–300`):
```cpp
fat = (bw * (0.14*age + 37.31*log(bw/(pow(ht,2.0))) - 103.94)/100.0)*(1-sex)
    + (bw * (0.14*age + 39.96*log(bw/(pow(ht,2.0))) - 102.01)/100.0)*sex;
```

**Verdict:** Match.

---

### Eq 5 — Total energy expenditure

**Paper:**
$$EE = K + \gamma_F F + \gamma_L L + \delta \cdot BW + TEF + AT + \eta_L \frac{dL}{dt} + \eta_F \frac{dF}{dt}$$

**Code** (`adult_weight.cpp:370–376`): The `R()` helper function substitutes EE into the energy partitioning equation and solves for $dL/dt$ algebraically, avoiding the need to compute EE explicitly:

```cpp
NumericVector R3 = K + delta*weight + TEF(t) + AT - TotalIntake(t) + roG*dG(t, G);
return (R3 + gammaL*L + gammaF*F) / (alfa1 + alfa2*F);
```

where `alfa1 = -(1 + etaL/roL)*C` and `alfa2 = -(1 + etaF/roF)`.

**Verdict:** Match.

---

### Eq 6 — Thermic effect of feeding

**Paper:**
$$TEF = \beta_{TEF} \cdot \Delta EI, \quad \beta_{TEF} = 0.1$$

**Code** (`adult_weight.cpp:311`):
```cpp
return betaTEF*deltaEI(t);
```

**Verdict:** Match.

---

### Eq 7 — Adaptive thermogenesis

**Paper:**
$$\tau_{AT} \frac{dAT}{dt} = \beta_{AT} \cdot \Delta EI - AT$$

**Code** (`adult_weight.cpp:321`):
```cpp
return (betaAT*deltaEI(t) - AT)*(1.0/tauAT);
```

**Verdict:** Match.

---

### Eq 8 — Physical activity parameter δ

**Paper:**
$$\delta = \left[(1 - \beta_{TEF}) \cdot PAL - 1\right] \cdot RMR / BW$$

**Code** (`adult_weight.cpp:287`):
```cpp
delta = ((1.0 - betaTEF)*PAL - 1.0)*rmr/bw;
```

**Verdict:** Match.

---

### RMR — Mifflin-St Jeor equation

**Paper (ref 16):**
$$RMR = 9.99 \cdot BW + 6.25 \cdot H_{cm} - 4.92 \cdot age + 5 \text{ (men)} \;/\; -161 \text{ (women)}$$

**Code** (`adult_weight.cpp:263`):
```cpp
rmr = (rmrbw*bw + rmrht*ht - rmrage*age + rmr_m)*(1-sex)
    + (rmrbw*bw + rmrht*ht - rmrage*age - rmr_f)*sex;
```

With `rmrht = 625.0` (height in meters, so $625 \times m = 6.25 \times cm$).

**Verdict:** Match.

---

## Constants

All values from the paper (in kJ) are correctly converted to kcal (×0.23900573614):

| Symbol    | Paper value    | Code variable | Code value | Correct |
|-----------|---------------|---------------|------------|---------|
| ρ_G       | 17.6 MJ/kg    | `roG`         | 4206.501   | Yes     |
| ρ_F       | 39.5 MJ/kg    | `roF`         | 9440.727   | Yes     |
| ρ_L       | 7.6 MJ/kg     | `roL`         | 1816.444   | Yes     |
| γ_F       | 13 kJ/kg/day  | `gammaF`      | 3.107      | Yes     |
| γ_L       | 92 kJ/kg/day  | `gammaL`      | 21.989     | Yes     |
| η_F       | 750 kJ/kg     | `etaF`        | 179.254    | Yes     |
| η_L       | 960 kJ/kg     | `etaL`        | 229.445    | Yes     |
| β_TEF     | 0.1           | `betaTEF`     | 0.1        | Yes     |
| β_AT      | 0.14          | `betaAT`      | 0.14       | Yes     |
| τ_AT      | 14 days       | `tauAT`       | 14.0       | Yes     |
| [Na]      | 3.22 mg/ml    | `Na`          | 3220       | Yes     |
| ζ_Na      | 3000 mg/L/d   | `zetaNa`      | 3000       | Yes     |
| ζ_CI      | 4000 mg/d     | `zetaCI`      | 4000       | Yes     |

---

## Discrepancies

None. All equations are aligned with the paper.

The glycogen energy flux term was previously missing the `roG` multiplier (commit `7476985` fixed this in both C++ and Python).

---

## Solver notes

- The ODE system is solved using **Runge-Kutta 4th order (RK4)**.
- AT, ECF, and glycogen are solved independently first at each time step; their half-step and full-step values are then used when solving for lean mass. This is an operator-splitting approach rather than a fully coupled solve — standard and reasonable for this system.
- Fat mass is derived from lean mass via the integrated Forbes relation rather than solved as a separate ODE.
- Body weight is reconstructed as $BW = F + L + ECF + 3.7 \cdot G$.

## References

1. Hall KD, Sacks G, Chandramohan D, et al. Quantification of the effect of energy imbalance on bodyweight. *Lancet*. 2011;378(9793):826–837.
2. Chow CC, Hall KD. The dynamics of human body weight change. *PLoS Comput Biol*. 2008;4(3):e1000045.
3. Hall KD. Predicting metabolic adaptation, body weight change, and energy intake in humans. *Am J Physiol Endocrinol Metab*. 2010;298(3):E449–E466.
4. Hall KD, Jordan PN. Modeling weight-loss maintenance to help prevent body weight regain. *Am J Clin Nutr*. 2008;88(6):1495–1503.
5. Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO. A new predictive equation for resting energy expenditure in healthy individuals. *Am J Clin Nutr*. 1990;51(2):241–247.
6. Jackson AS, Stanforth PR, Gagnon J, et al. The effect of sex, age and race on estimating percentage body fat from body mass index: The Heritage Family Study. *Int J Obes Relat Metab Disord*. 2002;26(6):789–96.
7. Forbes GB. Lean body mass-body fat interrelationships in humans. *Nutr Rev*. 1987;45(8):225–31.
