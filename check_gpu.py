import jax

print("JAX Backend:", jax.default_backend())
print("Devices found:", jax.devices())

try:
    x = jax.numpy.ones((1000, 1000))
    # Perform a matrix multiplication to trigger CuDNN/Cublas
    y = jax.numpy.dot(x, x) 
    print("Matrix multiplication successful on GPU.")
except Exception as e:
    print(f"GPU computation failed: {e}")