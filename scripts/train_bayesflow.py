import numpy as np
from models.bayesflow_model import BayesFlowModel

def prior(batch_size):
    return {'prior_draws': np.random.randint(0, 2, size=(batch_size, 50)).astype(np.float32)}

def simulator(prior_dict):
    x = prior_dict['prior_draws']
    return {
        'sim_data': x,
        'prior_draws': x 
    }

def simulator_wrapper(n_sim):
        prior_samples = prior(n_sim)
        sim_result = simulator(prior_samples)
        return sim_result

def train_bayesflow_model():
    model = BayesFlowModel()
    model.train(simulator_wrapper, prior)
