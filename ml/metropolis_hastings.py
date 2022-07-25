import numpy as np
import math
from scipy import stats



class MetropolisHastings(object):
    def __init__(self, markov_model, likelihood_func, prior, num_iterations):
        self.markov_model = markov_model
        self.likelihood_func = likelihood_func
        self.prior = prior
        self.num_iterations = num_iterations

    @staticmethod
    def __acceptance_rule(current: float, new: float) -> bool:
        return True if new > current else np.random.uniform(0, 1) < np.exp(new - current)

    def sample(self) -> (np.array, float):
        accepted = np.zeros(self.num_iterations+1)
        accepted_count = 0
        theta = 0.25
        accepted[0] = theta
        sigma = 0.2

        for i in range(self.num_iterations):
            new_theta = self.markov_model(theta, sigma)

            try:
                # Computes the prior for the current and next sample
                cur_prior = self.prior(theta)
                new_prior = self.prior(new_theta)
                if cur_prior > 0.0 and new_prior > 0.0:
                    # We use the logarithm value for the comparison to avoid underflow
                    cur_posterior = self.likelihood_func(theta) + np.log(cur_prior)
                    new_posterior = self.likelihood_func(new_theta) + np.log(new_prior)
                    if MetropolisHastings.__acceptance_rule(cur_posterior, new_posterior):
                        theta = new_theta
                        accepted_count += 1
                    accepted[i+1] = new_theta
            except ArithmeticError as e:
                print(f'Error {e}')
        return accepted, accepted_count/self.num_iterations


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
    def log_likelihood_normal(theta: float):
        return math.log(stats.binom(NormalMcMc.n, theta).pmf(NormalMcMc.h))

    @staticmethod
    def markov_model(theta: float, sigma: float) -> float:
        return theta + stats.norm(0.0, sigma).rvs()
