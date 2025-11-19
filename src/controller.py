import jax
import jax.numpy as jnp

def get_action(params, z, h):
    """
    Linear Controller.
    
    Steering: tanh -> [-1, 1]
    Gas:      (tanh + 1) / 2 -> [0, 1]
    Brake:    FORCED TO 0.0 (To prevent power-braking bug)
    """
    input_dim = 32 + 256
    output_dim = 3
    
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    b = params[w_end:]
    
    inp = jnp.concatenate([z, h])
    logits = jnp.dot(W, inp) + b
    
    # 1. Steering
    steer = jnp.tanh(logits[0])
    
    # 2. Gas
    gas = (jnp.tanh(logits[1]) + 1) / 2.0
    
    # 3. Brake -- HARDCODED FIX
    # We force brake to 0.0 so the car actually moves.
    brake = jnp.array(0.0)
    
    return jnp.stack([steer, gas, brake])