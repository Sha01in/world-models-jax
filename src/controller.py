import jax
import jax.numpy as jnp

def get_action_linear(params, z, h, action_dim=3):
    input_dim = z.shape[0] + h.shape[0]
    output_dim = action_dim
    
    w_end = input_dim * output_dim
    W = params[:w_end].reshape(output_dim, input_dim)
    b = params[w_end:]
    
    inp = jnp.concatenate([z, h])
    logits = jnp.dot(W, inp) + b
    
    return _process_logits(logits, action_dim)

def get_action_mlp(params, z, h, action_dim=3, hidden_dim=64):
    input_dim = z.shape[0] + h.shape[0]
    
    # Layer 1: Input -> Hidden
    w1_size = input_dim * hidden_dim
    b1_size = hidden_dim
    
    w1_end = w1_size
    b1_end = w1_end + b1_size
    
    W1 = params[:w1_end].reshape(hidden_dim, input_dim)
    b1 = params[w1_end:b1_end]
    
    # Layer 2: Hidden -> Output
    w2_size = hidden_dim * action_dim
    b2_size = action_dim
    
    w2_end = b1_end + w2_size
    b2_end = w2_end + b2_size
    
    W2 = params[b1_end:w2_end].reshape(action_dim, hidden_dim)
    b2 = params[w2_end:b2_end]
    
    inp = jnp.concatenate([z, h])
    
    # Forward
    hidden = jnp.tanh(jnp.dot(W1, inp) + b1)
    logits = jnp.dot(W2, hidden) + b2
    
    return _process_logits(logits, action_dim)

def _process_logits(logits, action_dim):
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

# Default alias for backward compatibility (though we should update callers)
get_action = get_action_linear