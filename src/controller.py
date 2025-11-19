import jax
import jax.numpy as jnp

def get_action(params, z, h):
    """
    Linear Controller with corrected action scaling.
    
    Steering: tanh -> [-1, 1]
    Gas:      (tanh + 1) / 2 -> [0, 1]
    Brake:    (tanh + 1) / 2 -> [0, 1]
    """
    input_dim = 32 + 256
    output_dim = 3
    
    # Unpack params
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    b = params[w_end:]
    
    inp = jnp.concatenate([z, h])
    logits = jnp.dot(W, inp) + b
    
    # --- CORRECTED ACTIVATIONS ---
    # 1. Steering (Index 0): Standard tanh (-1 to 1)
    steer = jnp.tanh(logits[0])
    
    # 2. Gas (Index 1): Scale to (0 to 1)
    gas = (jnp.tanh(logits[1]) + 1) / 2.0
    
    # 3. Brake (Index 2): Scale to (0 to 1)
    brake = (jnp.tanh(logits[2]) + 1) / 2.0
    
    return jnp.stack([steer, gas, brake])