import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler


class BidGPTSLearner(object):
    def __init__(self, arms, negative_probability_threshold):
        self.t = 0
        self.arms = arms
        self.negative_probability_threshold = negative_probability_threshold
        self.pulled_arms = []
        self.collected_rewards = []
        self.rounds_per_arm = np.zeros(len(self.arms))
        self.rewards_per_arm = [[] for _ in range(len(self.arms))]
        self.means = np.zeros(len(self.arms))
        self.sigmas = np.ones(len(self.arms)) * 1e3
        self.alpha = 10
        self.kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=9)
        self.scaler = StandardScaler()

    def update_observations(self, arm_idx, reward):
        self.t += 1
        self.pulled_arms.append(self.arms[arm_idx])
        self.collected_rewards.append(reward)
        self.rewards_per_arm[arm_idx].append(reward)
        self.rounds_per_arm[arm_idx] += 1

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        x = self.scaler.fit_transform(x)
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(self.scaler.transform(np.atleast_2d(self.arms).T), return_std=True)
        print(list(zip(self.means, self.sigmas)))
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def pull_arm(self):
        if len(np.argwhere(self.rounds_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))]

        samples = np.random.normal(self.means, self.sigmas)
        threshold_probs = np.array([norm(loc=loc, scale=scale).cdf(0) for (loc, scale) in zip(self.means, self.sigmas)])
        valid_arms_idx = np.argwhere(threshold_probs < self.negative_probability_threshold).reshape(-1)
        valid_arms = np.take(self.arms, valid_arms_idx).reshape(-1)
        samples = np.take(samples, valid_arms_idx).reshape(-1)
        if len(samples) == 0:
            raise Exception("All arms exceed negative probability threshold")
        return valid_arms[np.random.choice(np.argwhere(samples == samples.max()).reshape(-1))]

    def update(self, pulled_arm, reward):
        arm_idx = self.arms.index(pulled_arm)
        self.update_observations(arm_idx, reward)
        if len(self.collected_rewards) > 1:
            self.update_model()

    def get_optimal_arm(self):
        return self.arms[np.argmax(self.means)]