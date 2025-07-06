from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
from tensorflow.keras import Sequential, layers

class BayesFlowModel:
    def __init__(self):
        num_params = 50
        summary_net = Sequential([
            layers.Input(shape=(50,)), 
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu')
        ])

        self.posterior_net = InvertibleNetwork(num_params=num_params)
        self.amortizer = AmortizedPosterior(self.posterior_net, summary_net)

    def train(self, simulator, prior, n_epochs=5):
        trainer = Trainer(amortizer=self.amortizer, generative_model=simulator)
        trainer.train_online(
            prior=prior,
            n_simulations=1000,
            epochs=n_epochs,
            iterations_per_epoch=50,
            batch_size=32
        )

    def predict(self, x):
        return self.amortizer.sample(x, n_samples=100)
