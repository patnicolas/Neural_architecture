

class MCMC(object):
    from abc import abstractmethod

    @abstractmethod
    def sample(self, theta: float) -> float:
        pass