"""Portable inference for the authors' public Doom models.

Reference: hardmaru/WorldModelsExperiments, revision
fd982b9691a941b52c6addbde29bc801ca6202c8, doomrnn/{doomrnn,doomreal}.py.
The explicit TF gate order, restart input and NHWC dense ordering matter.
These supplied weights are a diagnostic control, not newly trained models.
"""

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from src.vae import VAE


def reference_preprocess(frame):
    """Reproduce SciPy imresize's RGB bytescale and the wrapper's final cast.

    imresize(float RGB) returns uint8, so the following legacy expression wraps
    modulo256 when cast to uint8. Normalized-float inversion is not equivalent.
    Pillow bilinear interpolation is retained; historical Pillow version is
    unknown and remains a protocol limitation until a legacy comparison.
    """
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Reference preprocessing needs a native uint8 RGB frame")
    if frame.shape[0] < 400:
        raise ValueError("Crop native observations before reducing resolution")
    image = frame[:400].astype(np.float64) / 255.0
    minimum = image.min()
    span = image.max() - minimum
    if span == 0:
        span = 1.0
    scaled = np.clip((image - minimum) * (255.0 / span), 0, 255)
    scaled = (scaled + 0.5).astype(np.uint8)
    resized = np.asarray(
        Image.fromarray(scaled).resize((64, 64), Image.Resampling.BILINEAR)
    )
    return ((1.0 - resized.astype(np.float64)) * 255).round().astype(np.uint8)


class ReferenceRNN(eqx.Module):
    kernel: jax.Array
    bias: jax.Array
    output_kernel: jax.Array
    output_bias: jax.Array
    latent_dim: int = eqx.field(static=True, default=64)
    hidden_size: int = eqx.field(static=True, default=512)
    num_gaussians: int = eqx.field(static=True, default=5)

    def __call__(self, inputs, hidden, restart=0.0):
        h, c = hidden
        h = jnp.where(restart > 0.5, 0.0, h)
        c = jnp.where(restart > 0.5, 0.0, c)
        joined = jnp.concatenate([inputs, jnp.asarray(restart).reshape(1), h])
        i, g, f, o = jnp.split(joined @ self.kernel + self.bias, 4)
        c = jax.nn.sigmoid(f + 1.0) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        output = h @ self.output_kernel + self.output_bias
        mixture = output[1:].reshape(self.latent_dim, 3 * self.num_gaussians)
        log_pi, mu, log_sigma = jnp.split(mixture, 3, axis=-1)
        log_pi = jax.nn.log_softmax(log_pi, axis=-1)
        return (log_pi.T, mu.T, log_sigma.T, jnp.zeros(1), output[:1]), (h, c)

    def init_state(self):
        return (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))


def checked_reference_arrays(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "download_manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Reference download is incomplete")
    result = {}
    for name in ("vae.json", "rnn.json", "controller.json", "initial_z.json"):
        path = directory / name
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["files"][name]["sha256"]
        ):
            raise ValueError("Pinned reference file changed")
        result[name] = json.loads(path.read_text())
    return result, manifest


def convert_vae(parameters):
    arrays = [np.asarray(p, np.float32) / 10000.0 for p in parameters]
    model = VAE(64, jax.random.PRNGKey(0), "paper")
    expected = []
    for a, b in zip((3, 32, 64, 128), (32, 64, 128, 256)):
        expected.extend([(4, 4, a, b), (b,)])
    expected.extend([(1024, 64), (64,), (1024, 64), (64,), (64, 1024), (1024,)])
    for k, a, b in zip((5, 5, 6, 6), (1024, 128, 64, 32), (128, 64, 32, 3)):
        expected.extend([(k, k, b, a), (b,)])
    if [a.shape for a in arrays] != expected:
        raise ValueError("Reference VAE tensor shapes differ")
    for index, layer in enumerate(model.encoder.layers):
        weight, bias = arrays[2 * index : 2 * index + 2]
        replacement = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            layer,
            (
                jnp.asarray(weight.transpose(3, 2, 0, 1)),
                jnp.asarray(bias[:, None, None]),
            ),
        )
        model = eqx.tree_at(lambda m: m.encoder.layers[index], model, replacement)
    for index, name in ((8, "mu_head"), (10, "logvar_head")):
        weight, bias = arrays[index : index + 2]
        # TensorFlow flattened HWC; Equinox flattens CHW.
        weight = weight.reshape(2, 2, 256, 64).transpose(3, 2, 0, 1).reshape(64, 1024)
        layer = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            getattr(model, name),
            (jnp.asarray(weight), jnp.asarray(bias)),
        )
        model = eqx.tree_at(lambda m: getattr(m, name), model, layer)
    linear = eqx.tree_at(
        lambda m: (m.weight, m.bias),
        model.decoder.linear,
        (jnp.asarray(arrays[12].T), jnp.asarray(arrays[13])),
    )
    model = eqx.tree_at(lambda m: m.decoder.linear, model, linear)
    for index, layer in enumerate(model.decoder.layers):
        weight, bias = arrays[14 + 2 * index : 16 + 2 * index]
        # Equinox uses a dilated-input correlation, hence reverse both axes.
        weight = weight.transpose(2, 3, 0, 1)[:, :, ::-1, ::-1].copy()
        replacement = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            layer,
            (jnp.asarray(weight), jnp.asarray(bias[:, None, None])),
        )
        model = eqx.tree_at(lambda m: m.decoder.layers[index], model, replacement)
    return model


def load_author_models(directory):
    payload, manifest = checked_reference_arrays(directory)
    vae = convert_vae(payload["vae.json"])
    arrays = [np.asarray(p, np.float32) / 10000.0 for p in payload["rnn.json"]]
    if [a.shape for a in arrays] != [(578, 2048), (2048,), (512, 961), (961,)]:
        raise ValueError("Reference RNN tensor shapes differ")
    rnn = ReferenceRNN(*(jnp.asarray(a) for a in arrays))
    controller = np.asarray(payload["controller.json"][0], np.float32)
    if controller.shape != (1088,) or not np.isfinite(controller).all():
        raise ValueError("Reference controller weights differ")
    return vae, rnn, jnp.asarray(controller), manifest
