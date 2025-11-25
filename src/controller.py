import jax
import jax.numpy as jnp

def get_action(params, z, h, action_dim=3):
    input_dim = z.shape[0] + h.shape[0]
    output_dim = action_dim
    
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    b = params[w_end:]
    
    inp = jnp.concatenate([z, h])
    logits = jnp.dot(W, inp) + b
    
    # If action_dim is 3, assume CarRacing (Steer, Gas, Brake)
    if action_dim == 3:
        # 1. Steering: Full Range [-1, 1]
        steer = jnp.tanh(logits[0])
        
        # 2. Gas: [0, 1]
        gas = jax.nn.sigmoid(logits[1])
        
        # 3. Brake: [0, 1]
        brake = jax.nn.sigmoid(logits[2])
        
        return jnp.stack([steer, gas, brake])
    else:
        # For Doom or generic, just use tanh for [-1, 1] range
        return jnp.tanh(logits)