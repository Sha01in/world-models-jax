import numpy as np
import jax
from train_vae import train_vae
import equinox as eqx
import os

def run_test():
    print("=== Running Smoke Test ===")
    
    # 1. Create Fake Data
    # Shape: (100 images, 3 channels, 64 height, 64 width)
    print("Generating fake random data...")
    fake_data = np.random.rand(100, 3, 64, 64).astype(np.float32)
    
    # 2. Run Training Loop
    print("Attempting to train VAE on fake data...")
    try:
        model = train_vae(fake_data, epochs=3)
        print("Training successful!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return

    # 3. Test Inference
    print("Testing inference on a single input...")
    dummy_input = jax.numpy.array(fake_data[0])
    key = jax.random.PRNGKey(42)
    
    try:
        recon, mu, logvar = model(dummy_input, key=key)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Recon shape: {recon.shape}") # Should be (3, 64, 64)
        print(f"Latent z shape: {mu.shape}")  # Should be (32,)
        
        assert recon.shape == (3, 64, 64), "Reconstruction shape mismatch!"
        assert mu.shape == (32,), "Latent shape mismatch!"
        print("Inference successful!")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # 4. Save Model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    print("Saving model to checkpoints/vae_test.eqx...")
    eqx.tree_serialise_leaves("checkpoints/vae_test.eqx", model)
    print("Done.")

if __name__ == "__main__":
    run_test()