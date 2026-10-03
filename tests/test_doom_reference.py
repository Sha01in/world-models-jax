"""Independent numeric checks for the supplied-model reference port."""

import ast
from pathlib import Path
import unittest

import jax.numpy as jnp
import numpy as np
from PIL import Image

from src.doom_reference import ReferenceRNN, reference_preprocess


def numpy_step(parameters, inputs, hidden, restart):
    kernel, bias, output_kernel, output_bias = parameters
    h, c = hidden
    if restart:
        h, c = np.zeros_like(h), np.zeros_like(c)
    joined = np.concatenate([inputs, [float(restart)], h]).astype(np.float32)
    i, g, f, o = np.split(joined @ kernel + bias, 4)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    c = sigmoid(f + 1) * c + sigmoid(i) * np.tanh(g)
    h = sigmoid(o) * np.tanh(c)
    return h @ output_kernel + output_bias, (h, c)


def numpy_encode(parameters, image):
    """NHWC valid cross-correlations and NHWC flatten, independent of the port."""
    h = np.asarray(image, np.float32)
    for index in range(4):
        kernel, bias = parameters[2 * index : 2 * index + 2]
        patches = np.lib.stride_tricks.sliding_window_view(h, (4, 4), axis=(0, 1))[
            ::2, ::2
        ]
        patches = patches.transpose(0, 1, 3, 4, 2)
        shape = patches.shape[:2]
        h = patches.reshape(
            -1, kernel.shape[0] * kernel.shape[1] * kernel.shape[2]
        ) @ kernel.reshape(-1, kernel.shape[-1])
        h = np.maximum(h.reshape(*shape, kernel.shape[-1]) + bias, 0)
    return h.reshape(-1) @ parameters[8] + parameters[9], h.reshape(-1) @ parameters[
        10
    ] + parameters[11]


def numpy_decode(parameters, z):
    """TensorFlow's transpose convolution as direct output scatter additions."""
    h = (z @ parameters[12] + parameters[13]).reshape(1, 1, 1024)
    for index in range(4):
        kernel, bias = parameters[14 + 2 * index : 16 + 2 * index]
        kh, kw, outputs, _ = kernel.shape
        height, width, _ = h.shape
        result = np.zeros(
            ((height - 1) * 2 + kh, (width - 1) * 2 + kw, outputs), np.float32
        )
        for y in range(kh):
            for x in range(kw):
                result[y : y + 2 * height : 2, x : x + 2 * width : 2] += (
                    h @ kernel[y, x].T
                )
        result += bias
        h = np.maximum(result, 0) if index < 3 else 1 / (1 + np.exp(-result))
    return h


class TestDoomReference(unittest.TestCase):
    def test_tf_gate_order_forget_offset_restart_and_mixture_layout(self):
        rng = np.random.default_rng(81)
        arrays = [
            rng.normal(0, 0.2, shape).astype(np.float32)
            for shape in ((7, 8), (8,), (2, 19), (19,))
        ]
        model = ReferenceRNN(
            *(jnp.asarray(a) for a in arrays),
            latent_dim=3,
            hidden_size=2,
            num_gaussians=2,
        )
        expected_state = (
            np.array([0.2, -0.4], np.float32),
            np.array([0.8, 0.9], np.float32),
        )
        actual_state = tuple(jnp.asarray(a) for a in expected_state)
        for restart in (1, 0, 0, 1, 0):
            inputs = rng.normal(size=4).astype(np.float32)
            expected, expected_state = numpy_step(
                arrays, inputs, expected_state, restart
            )
            prediction, actual_state = model(
                jnp.asarray(inputs), actual_state, float(restart)
            )
            for actual, wanted in zip(actual_state, expected_state):
                np.testing.assert_allclose(actual, wanted, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(
                prediction[-1], expected[:1], rtol=2e-6, atol=2e-7
            )
            mixture = expected[1:].reshape(3, 6)
            np.testing.assert_allclose(
                prediction[1].T, mixture[:, 2:4], rtol=2e-6, atol=2e-7
            )
            np.testing.assert_allclose(
                prediction[2].T, mixture[:, 4:], rtol=2e-6, atol=2e-7
            )
            np.testing.assert_allclose(np.exp(prediction[0]).sum(axis=0), 1, atol=2e-7)

    def test_native_crop_and_uint8_wrap_are_not_normalized_inversion(self):
        frame = np.broadcast_to(
            np.arange(256, dtype=np.uint8)[None, :, None], (480, 256, 3)
        ).copy()
        original = reference_preprocess(frame)
        frame[400:] = 177
        np.testing.assert_array_equal(reference_preprocess(frame), original)
        ramp = np.array([0, 1, 254, 255], np.uint8)
        np.testing.assert_array_equal(
            ((1.0 - ramp) * 255).round().astype(np.uint8), [255, 0, 253, 254]
        )
        with self.assertRaisesRegex(ValueError, "native"):
            reference_preprocess(np.zeros((64, 64, 3), np.uint8))

    def test_preprocessing_matches_pinned_original_scipy_functions(self):
        root = Path("artifacts/doom_reference_round5/fd982b9")
        scipy_path = root / "legacy/scipy_pilutil.py"
        if not scipy_path.exists():
            self.skipTest("Optional pinned legacy source is local to the audit")

        class LegacyPillow:
            @staticmethod
            def isImageType(value):
                # Pillow removed this helper; its check was isinstance(Image).
                return isinstance(value, Image.Image)

            def __getattr__(self, name):
                return getattr(Image, name)

        namespace = {name: getattr(np, name) for name in dir(np)}
        namespace.update(
            numpy=np,
            np=np,
            Image=LegacyPillow(),
            SCREEN_Y=64,
            SCREEN_X=64,
            _errstr="image mode error",
        )
        functions = []

        class LegacyBytesAlias(ast.NodeTransformer):
            def visit_Attribute(self, node):
                node = self.generic_visit(node)
                # ndarray.tostring was an alias for tobytes, removed in NumPy 2.
                if node.attr == "tostring":
                    node.attr = "tobytes"
                return node

        for node in ast.parse(scipy_path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "bytescale",
                "toimage",
                "fromimage",
                "imresize",
            }:
                node.decorator_list = []
                functions.append(LegacyBytesAlias().visit(node))
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), str(scipy_path), "exec"
            ),
            namespace,
        )
        namespace["resize"] = namespace["imresize"]
        frame_function = next(
            node
            for node in ast.parse((root / "source/doomreal.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_process_frame"
        )

        # The removed np.float alias was Python float, equivalent to float64 here.
        class LegacyNumpy:
            float = float

            def __getattr__(self, name):
                return getattr(np, name)

        namespace["np"] = LegacyNumpy()
        exec(
            compile(
                ast.Module(body=[frame_function], type_ignores=[]),
                "original_frame",
                "exec",
            ),
            namespace,
        )
        rng = np.random.default_rng(82)
        for frame in (
            rng.integers(20, 230, (480, 640, 3), dtype=np.uint8),
            rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            np.full((480, 640, 3), 77, np.uint8),
        ):
            np.testing.assert_array_equal(
                reference_preprocess(frame), namespace["_process_frame"](frame)
            )


if __name__ == "__main__":
    unittest.main()
