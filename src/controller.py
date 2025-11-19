import jax
import jax.numpy as jnp

def get_action(params, z, h):
    """
    Linear Controller: a = tanh( W * [z, h] + b )
    
    params: Flat vector containing weights and biases
    z: Latent vector (32,)
    h: Hidden state (256,)
    """
    # Input size = 32 (z) + 256 (h) = 288
    # Output size = 3 (steer, gas, brake)
    input_dim = 32 + 256
    output_dim = 3
    
    # Unpack flat params
    # Weight matrix: (3, 288)
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    
    # Bias vector: (3,)
    b = params[w_end:]
    
    # Concatenate inputs
    inp = jnp.concatenate([z, h])
    
    # Linear pass
    logits = jnp.dot(W, inp) + b
    
    # Activation
    action = jnp.tanh(logits)
    return action