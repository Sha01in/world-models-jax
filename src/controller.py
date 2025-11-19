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
    
    # 1. Steering: Full Range [-1, 1]
    steer = jnp.tanh(logits[0])
    
    # 2. Gas: [0, 1]
    gas = jax.nn.sigmoid(logits[1])
    
    # 3. Brake: [0, 1]
    brake = jax.nn.sigmoid(logits[2])
    
    return jnp.stack([steer, gas, brake])