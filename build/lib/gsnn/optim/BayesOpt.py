import numpy as np
from ax import ParameterType
import torch
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core.objective import MultiObjective, Objective
from ax.metrics.noisy_function import GenericNoisyFunctionMetric

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.acquisition.logei import qLogNoisyExpectedImprovement



# https://ax.dev/versions/latest/tutorials/saasbo_nehvi.html
class BayesOptAgent:
    def __init__(
        self, env, n_actions, warmup=10, action_labels=None, 
        verbose=True, suppress_warnings=True, **kwargs
    ):
        if suppress_warnings:
            import logging
            import warnings
            logging.getLogger('ax').setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=UserWarning)

        self.action_labels = action_labels
        self.env = env
        self.n_actions = n_actions
        self.verbose = verbose
        self.iteration = 0
        self.rewards = []
        self.best_actions = None
        self.best_rewards = None

        # Define the surrogate model using SaasFullyBayesianSingleTaskGP
        surrogate = Surrogate(
            botorch_model_class=SaasFullyBayesianSingleTaskGP,
            mll_options={
                "num_samples": 256,   # Number of MCMC samples
                "warmup_steps": 512,  # Number of warm-up steps in MCMC
            },
        )

        # Define the generation strategy with the surrogate and acquisition function
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=warmup,  # Number of initial random samples
                    model_kwargs={"seed": 999},  # For reproducibility
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # Continue indefinitely
                    model_kwargs={
                        "surrogate": surrogate,
                        "botorch_acqf_class": qLogNoisyExpectedImprovement,
                    },
                ),
            ]
        )

        # Create the AxClient with the custom generation strategy
        self.client = AxClient(verbose_logging=verbose, generation_strategy=gs)

        # Create the experiment with the specified objective
        self.client.create_experiment(
            name="gsnn_graph_optim",
            parameters=[
                {
                    "name": f"a{i}",
                    "type": "choice",
                    "values": [0, 1],
                    "value_type": "int",
                    "is_ordered": True,
                    "sort_values": False,
                }
                for i in range(self.n_actions)
            ],
            objectives={f"reward": ObjectiveProperties(minimize=False)},
        )

    def evaluate(self, parameters):
        action = torch.tensor(
            [parameters[f"a{i}"] for i in range(self.n_actions)], dtype=torch.long
        )
        outputs = self.env.run(action)
        self.rewards.append(outputs.mean())
        metrics = {f"reward": outputs.mean()}
        return metrics
    
    def record_best_action(self):
        
        best_parameters, values = self.client.get_best_parameters()

        self.best_action = [best_parameters[f"a{i}"] for i in range(self.n_actions)]
        self.best_reward = values

    def ensure_numpy(self, x): 
        if isinstance(x, torch.Tensor): 
            return x.detach().cpu().numpy()
        return x

    def print_progress_(self):

        if self.action_labels is not None: 
            best_action = self.ensure_numpy(self.best_action)
            action_labels = self.ensure_numpy(self.action_labels)
            acc =  (1.*(best_action == action_labels)).mean()
            print(f'\t\t\t --> iter: {self.iteration} || best action acc: {acc:.3f} || last reward: {self.rewards[-1]:.3f}')

        else:
            print(f'\t\t\t --> iter: {self.iteration} || last reward: {self.rewards[-1]:.3f}')


    def step(self):
        # Get the next trial parameters
        parameters, trial_index = self.client.get_next_trial()
        # Evaluate and complete the trial
        self.client.complete_trial(
            trial_index=trial_index, raw_data=self.evaluate(parameters)
        )
        
        self.record_best_action()

        self.iteration += 1

        self.print_progress_()


    def get_best_action(self):
        if self.best_actions is not None:
            return self.best_actions
        else:
            # Return default action if none have been tried yet
            return [np.zeros(self.n_actions, dtype=int)]