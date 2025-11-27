import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

class CMAESState(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray
    sigma: jnp.ndarray
    p_sigma: jnp.ndarray
    p_c: jnp.ndarray
    gen: jnp.ndarray
    key: jnp.ndarray

class CMA_ES:
    def __init__(self, num_params: int, pop_size: int, sigma_init: float = 0.1):
        self.num_params = num_params
        self.pop_size = pop_size
        self.sigma_init = sigma_init
        
        # Convert to float for stability in calculations
        n = float(num_params)

        # Strategy parameters
        self.mu = pop_size // 2
        self.weights = jnp.log(self.mu + 0.5) - jnp.log(jnp.arange(1, self.mu + 1))
        self.weights = self.weights / jnp.sum(self.weights)
        self.mueff = 1.0 / jnp.sum(self.weights**2)
        
        # Time constants
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, jnp.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        
        self.chiN = jnp.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    def init(self, key: jnp.ndarray, mean_init: jnp.ndarray = None) -> CMAESState:
        if mean_init is None:
            mean_init = jnp.zeros(self.num_params)
            
        return CMAESState(
            mean=mean_init,
            cov=jnp.eye(self.num_params),
            sigma=jnp.array(self.sigma_init),
            p_sigma=jnp.zeros(self.num_params),
            p_c=jnp.zeros(self.num_params),
            gen=jnp.array(0),
            key=key
        )

    def ask(self, state: CMAESState) -> Tuple[jnp.ndarray, CMAESState]:
        key, subkey = jax.random.split(state.key)
        
        # Eigendecomposition
        # For stability, enforce symmetry
        cov = (state.cov + state.cov.T) / 2.0
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        
        # Ensure positive definite
        eigvals = jnp.maximum(eigvals, 1e-16)
        
        # Sample
        z = jax.random.normal(subkey, (self.pop_size, self.num_params))
        y = jnp.dot(z, jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T)
        candidates = state.mean + state.sigma * y
        
        new_state = state._replace(key=key)
        return candidates, new_state

    def tell(self, state: CMAESState, candidates: jnp.ndarray, fitness: jnp.ndarray) -> CMAESState:
        # Sort by fitness (ascending - minimization)
        idx = jnp.argsort(fitness)
        sorted_candidates = candidates[idx]
        
        # Selection
        best_candidates = sorted_candidates[:self.mu]
        
        # Update mean
        mean_old = state.mean
        mean_new = jnp.dot(self.weights, best_candidates)
        
        # Evolution paths
        y = (mean_new - mean_old) / state.sigma
        
        # Inverse sqrt of covariance matrix (C^-1/2)
        # Using eigendecomposition from ask step would be more efficient if cached, 
        # but for simplicity recomputing or approximating here.
        # Actually, let's use the z vectors if possible, but we only have candidates.
        # Reconstruct z-like vectors:
        # z_w = C^-1/2 * y
        # For full CMA, we need proper updates.
        
        cov = (state.cov + state.cov.T) / 2.0
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        eigvals = jnp.maximum(eigvals, 1e-16)
        inv_sqrt_cov = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T
        
        z_w = jnp.dot(inv_sqrt_cov, y)
        
        # Update p_sigma
        p_sigma_new = (1 - self.cs) * state.p_sigma + \
                      jnp.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
        
        # H_sigma indicator
        norm_p_sigma = jnp.linalg.norm(p_sigma_new)
        h_sigma = (norm_p_sigma / jnp.sqrt(1 - (1 - self.cs)**(2 * (state.gen + 1))) < \
                   (1.4 + 2 / (self.num_params + 1)) * self.chiN)
        h_sigma = h_sigma.astype(jnp.float32)
        
        # Update p_c
        p_c_new = (1 - self.cc) * state.p_c + \
                  h_sigma * jnp.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        
        # Update Covariance
        # Rank-1 update
        rank1 = jnp.outer(p_c_new, p_c_new)
        
        # Rank-mu update
        # y_k = (x_k - m_old) / sigma
        y_k = (best_candidates - mean_old) / state.sigma
        rank_mu = jnp.dot(y_k.T, (y_k * self.weights[:, None]))
        
        cov_new = (1 - self.c1 - self.cmu) * state.cov + \
                  self.c1 * (rank1 + (1 - h_sigma) * self.cc * (2 - self.cc) * state.cov) + \
                  self.cmu * rank_mu
                  
        # Update Sigma
        sigma_new = state.sigma * jnp.exp((self.cs / self.damps) * (norm_p_sigma / self.chiN - 1))
        
        return CMAESState(
            mean=mean_new,
            cov=cov_new,
            sigma=sigma_new,
            p_sigma=p_sigma_new,
            p_c=p_c_new,
            gen=state.gen + 1,
            key=state.key
        )
