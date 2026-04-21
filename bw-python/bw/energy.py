"""Energy intake interpolation methods."""
import numpy as np


class EnergyBuilder:
    """Interpolate discrete energy measurements to daily time series."""

    METHODS = {"linear", "exponential", "logarithmic", "stepwise_l", "stepwise_r", "brownian"}

    def build(self, energy, time, method="brownian"):
        """Interpolate energy measurements to daily values.

        Parameters
        ----------
        energy : array-like, shape (n_individuals, n_measurements)
            Energy measurements at discrete time points.
        time : array-like, shape (n_measurements,)
            Measurement times. Must start at 0, strictly increasing integers.
        method : str
            Interpolation method.

        Returns
        -------
        np.ndarray, shape (n_individuals, n_days + 1)
            Daily energy values from day 0 to day time[-1].
        """
        energy = np.atleast_2d(np.asarray(energy, dtype=float))
        time = np.asarray(time, dtype=float)

        self._validate(energy, time, method)

        days = int(np.floor(time[-1]))
        n_ind = energy.shape[0]
        result = np.zeros((n_ind, days + 1))

        method_lower = method.lower()

        if method_lower == "brownian":
            return self._brownian(energy, time, n_ind, days)

        j = 0  # index into time segments
        for i in range(days):
            t_j = time[j]
            t_j1 = time[j + 1]
            dt = t_j1 - t_j

            if method_lower == "linear":
                result[:, i] = ((energy[:, j + 1] - energy[:, j]) / dt * (i - t_j)
                                + energy[:, j])
            elif method_lower == "stepwise_l":
                result[:, i] = energy[:, j]
            elif method_lower == "stepwise_r":
                result[:, i] = energy[:, j + 1]
            elif method_lower == "exponential":
                K = 5000.0
                result[:, i] = (np.exp((np.log(energy[:, j + 1] - energy[:, j] + K) - np.log(K))
                                       / dt * (i - t_j) + np.log(K))
                                - K + energy[:, j])
            elif method_lower == "logarithmic":
                result[:, i] = (1000.0 * np.log(
                    (np.exp((energy[:, j + 1] - energy[:, j]) / 1000.0) - 1)
                    / dt * (i - t_j) + 1)
                    + energy[:, j])

            # Advance segment
            if i + 1 >= time[j + 1]:
                j += 1

        # Last day = last measurement
        result[:, -1] = energy[:, -1]
        return result

    def _brownian(self, energy, time, n_ind, days):
        result = np.zeros((n_ind, days + 1))
        for j in range(len(time) - 1):
            T = time[j + 1]
            t = time[j]
            span = int(T - t)

            # Simulate Brownian path W
            W = np.zeros((n_ind, span + 1))
            for i in range(1, span + 1):
                W[:, i] = W[:, i - 1] + np.random.randn(n_ind)

            # Brownian bridge
            for i in range(span + 1):
                idx = int(t) + i
                if idx <= days:
                    result[:, idx] = (energy[:, j] * (span - i) / span
                                      + energy[:, j + 1] * i / span
                                      + W[:, i] - (i / span) * W[:, span])
        return result

    def _validate(self, energy, time, method):
        if method.lower() not in self.METHODS:
            raise ValueError(f"Unknown interpolation method '{method}'. "
                             f"Must be one of: {', '.join(sorted(self.METHODS))}")
        if time[0] != 0:
            raise ValueError("time must start at 0")
        if len(time) < 2:
            raise ValueError("time must have at least 2 elements")
        if not np.all(np.diff(time) > 0):
            raise ValueError("time must be strictly increasing")
        if energy.shape[1] != len(time):
            raise ValueError(f"energy has {energy.shape[1]} columns but time has {len(time)} elements")
