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
    
    # 2. Gas: Scaled to (0.0 to 0.6) <--- TRAINING WHEELS
    # Standard sigmoid is (0 to 1). We multiply by 0.6.
    gas = (jnp.tanh(logits[1]) + 1) / 2.0 * 0.6
    
    # 3. Brake: Forced to 0.0
    brake = jnp.array(0.0)
    
    return jnp.stack([steer, gas, brake])