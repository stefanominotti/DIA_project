import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from PriceBidLearner import PriceBidLearner


class PriceBidGPTSLearner(PriceBidLearner):
    def update(self, pulled_arm, customers, returns=[]):
        super.update(pulled_arm, customers, returns=returns)

        if len(self.pulled_arms) > 1:
            self.daily_clicks_means, self.daily_clicks_sigmas = self.gp_regression(self.collected_daily_clicks_per_arm)
    
    def gp_regression(self, samples_per_arm):
        alpha = 10
        kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9)
        scaler = StandardScaler()
        x = []
        y = []
        for idx, arm in enumerate(samples_per_arm):
            x.extend([self.arms[idx] for _ in range(len(arm))])
            y.extend(arm)
        x = np.atleast_2d(x).T
        x = scaler.fit_transform(x)
        gp.fit(x, y)
        means, sigmas = gp.predict(scaler.transform(np.atleast_2d(self.arms).T), return_std=True)
        sigmas = np.maximum(sigmas, 1e-2)
        return means, sigmas