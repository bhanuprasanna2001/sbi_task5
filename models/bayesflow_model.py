from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense

class BayesFlowModel:
    def __init__(self):
        num_params = 50

        input_layer = Input(shape=(50,), name="sim_data")
        x = Dense(100, activation='relu')(input_layer)
        x = Dense(50, activation='relu')(x)
        self.summary_net = Model(inputs=input_layer, outputs=x, name="summary_net")

        self.posterior_net = InvertibleNetwork(num_params=num_params)

        self.amortizer = AmortizedPosterior(self.posterior_net, self.summary_net)

    def train(self, simulator, prior, n_epochs=5):
        trainer = Trainer(amortizer=self.amortizer, generative_model=simulator)
        trainer.train_online(
            prior=prior,
            n_simulations=1000,
            epochs=n_epochs,
            iterations_per_epoch=50,
            batch_size=32
        )

    def save(self):
        self.summary_net.save("saved_models/summary_net.keras")
        self.posterior_net.save_weights("saved_models/posterior_net.weights.h5")

