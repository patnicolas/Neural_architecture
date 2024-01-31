__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import numpy as np
import math
from scipy import stats


class ProposalDistribution(object):
    from abc import abstractmethod

    @abstractmethod
    def prior(self, theta: float) -> float:
        pass

    def log_likelihood(self, theta: float) -> float:
        pass

    def update_step(self, theta: float, sigma_diff: float) -> float:
        pass

    def posterior(self, theta: float, prior_value: float) -> float:
        pass


class ProposalBeta(ProposalDistribution):
    pi_2_inv = np.sqrt(2 * np.pi)

    def __init__(self, alpha: int, beta: int, num_trials: int, h: int):
        self.alpha = alpha
        self.beta = beta
        self.num_trials = num_trials
        self.h = h

    def prior(self, theta: float) -> float:
        x = stats.beta(self.alpha, self.beta).pdf(theta)
        return x if x > 0.0 else 1e-5

    def log_likelihood(self, theta: float) -> float:
        """
            Compute the Probability Mass Function for the binomial distribution C(num_trials, theta)
            :param theta Value of parameter to evaluate
            :return log of the probability mass distribution of the beta distribution
        """
        return math.log(stats.binom(self.num_trials, theta).pmf(self.h))

    def update_step(self, theta: float, sigma_diff: float) -> float:
        return theta + stats.norm(0.0, sigma_diff).rvs()

    def posterior(self, theta: float, prior_value: float) -> float:
        return self.log_likelihood(theta) + np.log(prior_value)
