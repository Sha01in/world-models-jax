import jax
import jax.numpy as jnp
import equinox as eqx

class MDNRNN(eqx.Module):
    lstm: eqx.nn.LSTMCell
    mdn_head: eqx.nn.Linear
    latent_dim: int
    hidden_size: int
    num_gaussians: int

    def __init__(self, latent_dim=32, action_dim=3, hidden_size=256, num_gaussians=5, key=None):
        k1, k2 = jax.random.split(key, 2)
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        
        # Input: z (32) + action (3) = 35
        self.lstm = eqx.nn.LSTMCell(latent_dim + action_dim, hidden_size, key=k1)
        
        # Output: Parameters for Mixture Density Network
        # For each Gaussian, we need:
        # 1. log_pi (weight) -> 1 per mixture
        # 2. mu (mean)       -> latent_dim per mixture
        # 3. log_sigma (std) -> latent_dim per mixture
        # Total output = num_gaussians * (2 * latent_dim + 1)
        out_size = num_gaussians * (2 * latent_dim + 1)
        self.mdn_head = eqx.nn.Linear(hidden_size, out_size, key=k2)

    def __call__(self, input_concat, hidden_state):
        # Step the LSTM
        hidden_state = self.lstm(input_concat, hidden_state)
        h, c = hidden_state
        
        # Get MDN params
        output = self.mdn_head(h)
        
        # Reshape: (num_gaussians, 2*latent_dim + 1)
        output = jnp.reshape(output, (self.num_gaussians, -1))
        
        # Split parameters
        log_pi = output[:, :1]                      # (5, 1)
        mu = output[:, 1 : 1 + self.latent_dim]     # (5, 32)
        log_sigma = output[:, 1 + self.latent_dim:] # (5, 32)
        
        # Normalize mixture weights (log_softmax ensures exp(log_pi) sums to 1)
        log_pi = jax.nn.log_softmax(log_pi, axis=0)
        
        # We allow log_sigma to be whatever; we will exp() it in the loss function
        
        return (log_pi, mu, log_sigma), hidden_state

    def init_state(self):
        return (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))