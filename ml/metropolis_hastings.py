__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

import numpy as np
import math
from scipy import stats
from typing import Callable

"""
    Implementation of Metropolis-Hastings Monte Carlo Markov Chain
    :param markov_model Markov model of transitional probabilities 
    :param likelihood_function Likelihood associated with a proposed distribution
    :param prior Class prior probability associated with the proposed distribution
    :param num_iterations Number of iterations for the random walk
    :param burn_in_ratio Percentage of number of iterations dedicated to burn-in steps
    :param sigma_delta Covariance or standard deviation used for each step theta -> theta_star
    @author Patrick Nicolas
"""


class MetropolisHastings(object):

    def __init__(self,
                 markov_model: Callable[[float, float], float],
                 likelihood_func: Callable[[float], float],
                 prior: Callable[[float], float],
                 num_iterations: int,
                 burn_in_ratio: float,
                 sigma_delta: float = 0.2):
        assert 1 < num_iterations < 10000, f'Number of iterations {num_iterations} is out of bounds'

        self.markov_model = markov_model
        self.likelihood_func = likelihood_func
        self.prior = prior
        self.num_iterations = num_iterations
        self.sigma_delta = sigma_delta
        self.burn_ins = int(num_iterations*burn_in_ratio)

    def sample(self, theta_0: float) -> (np.array, float):
        theta_walk = np.zeros(self.num_iterations + 1)
        accepted_count = 0
        theta = theta_0    # 0.25
        theta_walk[0] = theta

        for i in range(self.num_iterations):
            theta_star = self.markov_model(theta,self. sigma_delta)

            try:
                # Computes the prior for the current and next sample
                cur_prior = self.prior(theta)
                new_prior = self.prior(theta_star)

                # We only consider positive and non-null prior probabilities
                if cur_prior > 0.0 and new_prior > 0.0:
                    # We use the logarithm value for the comparison to avoid underflow
                    cur_log_posterior = self.__posterior(theta, cur_prior)
                    new_log_posterior = self.__posterior(theta_star, new_prior)

                    # Apply the selection criteria
                    if MetropolisHastings.__acceptance_rule(cur_log_posterior, new_log_posterior):
                        theta = theta_star
                        if i > self.burn_ins:
                            accepted_count += 1
                            theta_walk[i + 1] = theta_star
                    else:
                        if i > self.burn_ins:
                            theta_walk[i + 1] = theta_walk[i]

            except ArithmeticError as e:
                print(f'Error {e}')
        return theta_walk, float(accepted_count) / self.num_iterations

        # --------------  Supporting methods -----------------------

    def __posterior(self, theta: float, prior: float) -> float:
        return  self.likelihood_func(theta) + np.log(prior)

    @staticmethod
    def __acceptance_rule(current: float, new: float) -> bool:
        residual = new - current
        return True if new > current else np.random.uniform(0, 1) < np.exp(residual)


class NormalMcMc(object):
    pi_2_inv = np.sqrt(2 * np.pi)
    a = 12
    b = 10
    n = 96
    h = 10

    @staticmethod
    def prior(theta: float) -> float:
        x = stats.beta(NormalMcMc.a, NormalMcMc.b).pdf(theta)
        return x if x > 0.0 else 1e-5

    @staticmethod
    def log_likelihood(theta: float) -> float:
        return math.log(stats.binom(NormalMcMc.n, theta).pmf(NormalMcMc.h))

    @staticmethod
    def markov_model(theta: float, sigma_diff: float) -> float:
        return theta + stats.norm(0.0, sigma_diff).rvs()

    @staticmethod
    def posterior(theta: float, prior: float) -> float:
        return NormalMcMc.log_likelihood(theta) + np.log(prior)