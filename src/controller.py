import jax
import jax.numpy as jnp

def get_action(params, z, h):
    input_dim = 32 + 256
    output_dim = 3
    
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    b = params[w_end:]
    
    inp = jnp.concatenate([z, h])
    logits = jnp.dot(W, inp) + b
    
    # 1. Steering: Normal (-1 to 1)
    steer = jnp.tanh(logits[0])
    
    # 2. Gas: CLAMPED [0.4 to 0.8]
    # (tanh+1)/2 is 0..1. 
    # We scale it to 0..0.4 and add 0.4 base.
    # This ensures the car NEVER stops moving.
    raw_gas = (jnp.tanh(logits[1]) + 1) / 2.0
    gas = raw_gas * 0.4 + 0.4
    
    # 3. Brake: Forced to 0.0 to prevent conflicts
    brake = jnp.array(0.0)
    
    return jnp.stack([steer, gas, brake])