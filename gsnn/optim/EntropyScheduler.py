
class EntropyScheduler:
    """
    Class to handle entropy decay at defined intervals.

    Parameters:
    - initial_entropy (float): Initial entropy value.
    - decay_rate (float): Factor by which the entropy is decayed.
    - decay_interval (int): Number of iterations between each decay step.
    - min_entropy (float): Minimum entropy value to maintain after decay.
    """
    def __init__(self, initial_entropy, decay_rate=0.5, decay_interval=100, min_entropy=1e-4, verbose=True):
        self.entropy = initial_entropy
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.min_entropy = min_entropy
        self.iteration = 0
        self.verbose = verbose

    def step(self):
        """
        Decay the entropy based on the decay interval and rate, respecting the minimum entropy threshold.
        """
        # Increment iteration count
        self.iteration += 1

        # Check if it's time to decay the entropy
        if self.iteration % self.decay_interval == 0:
            if self.verbose: 
                print()
                print(f'decaying entropy: {self.get_entropy():5f}->{max(self.entropy * self.decay_rate, self.min_entropy):.5f}')
            self.entropy = max(self.entropy * self.decay_rate, self.min_entropy)

    def get_entropy(self):
        """
        Get the current entropy value.
        """
        return self.entropy