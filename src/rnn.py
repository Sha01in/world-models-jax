import jax
import jax.numpy as jnp
import equinox as eqx

class MDNRNN(eqx.Module):
    lstm: eqx.nn.LSTMCell
    mdn_head: eqx.nn.Linear
    reward_head: eqx.nn.Linear  # <--- New
    done_head: eqx.nn.Linear    # <--- New
    latent_dim: int
    hidden_size: int
    num_gaussians: int

    def __init__(self, latent_dim=32, action_dim=3, hidden_size=256, num_gaussians=5, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        
        # Input: z (32) + action (3) = 35
        self.lstm = eqx.nn.LSTMCell(latent_dim + action_dim, hidden_size, key=k1)
        
        # 1. MDN Head (Next z)
        out_size = num_gaussians * (2 * latent_dim + 1)
        self.mdn_head = eqx.nn.Linear(hidden_size, out_size, key=k2)
        
        # 2. Reward Head (Scalar prediction)
        self.reward_head = eqx.nn.Linear(hidden_size, 1, key=k3)
        
        # 3. Done Head (Binary logit)
        self.done_head = eqx.nn.Linear(hidden_size, 1, key=k4)

    def __call__(self, input_concat, hidden_state):
        # Step LSTM
        hidden_state = self.lstm(input_concat, hidden_state)
        h, c = hidden_state
        
        # A. MDN Prediction
        mdn_out = self.mdn_head(h)
        mdn_out = jnp.reshape(mdn_out, (self.num_gaussians, -1))
        log_pi = jax.nn.log_softmax(mdn_out[:, :1], axis=0)
        mu = mdn_out[:, 1 : 1 + self.latent_dim]
        log_sigma = mdn_out[:, 1 + self.latent_dim:]
        
        # B. Reward Prediction (Linear -> Scalar)
        reward = self.reward_head(h) # Shape (1,)
        
        # C. Done Prediction (Linear -> Logit)
        done_logit = self.done_head(h) # Shape (1,)
        
        return (log_pi, mu, log_sigma, reward, done_logit), hidden_state

    def init_state(self):
        return (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))