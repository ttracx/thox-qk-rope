"""Oracle tests for the thox-qk-rope v1 numerical contract.

The oracle below is written out from the formulas, not imported from the
reference, so a regression in reference.py fails here.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

from thox_qk_rope.reference import apply_rope, qk_norm, qk_norm_rope  # noqa: E402


def oracle_rms(x, weight, eps=1e-6):
    xf = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf * (1.0 / np.sqrt(var + np.float32(eps)))
    cast = normed.astype(x.dtype, copy=False)
    out_dtype = np.result_type(w.dtype, cast.dtype)
    return (w.astype(out_dtype) * cast.astype(out_dtype)).astype(out_dtype)


def oracle_rope(x, cos, sin):
    """NeoX rotate-half, with the head-broadcast rule spelled out."""
    x = np.asarray(x)
    cos = np.asarray(cos)
    sin = np.asarray(sin)
    rotary = cos.shape[-1]
    if cos.ndim == x.ndim - 1:
        cos = cos[..., None, :]
        sin = sin[..., None, :]
    rot = np.asarray(x[..., :rotary], dtype=np.float32)
    half = rotary // 2
    first, second = rot[..., :half], rot[..., half:]
    c = np.asarray(cos, dtype=np.float32)
    s = np.asarray(sin, dtype=np.float32)
    c1, c2 = c[..., :half], c[..., half:]
    s1, s2 = s[..., :half], s[..., half:]
    out1 = first * c1 - second * s1
    out2 = second * c2 + first * s2
    out = np.concatenate((out1, out2), axis=-1)
    out_dtype = np.result_type(x.dtype, cos.dtype)
    out = out.astype(out_dtype)
    if rotary == x.shape[-1]:
        return out
    return np.concatenate((out, x[..., rotary:].astype(out_dtype)), axis=-1)


class ContractTests(unittest.TestCase):
    def test_rotation_of_basis(self):
        # (1, 0) with cos=0, sin=1 on both halves is a +90 degree rotate-half step.
        x = np.array([1.0, 0.0], dtype=np.float32)
        cos = np.array([0.0, 0.0], dtype=np.float32)
        sin = np.array([1.0, 1.0], dtype=np.float32)
        y = apply_rope(x, cos, sin)
        np.testing.assert_allclose(y, [0.0, 1.0], atol=1e-6)

    def test_qk_norm_matches_oracle(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, 4, 8)).astype(np.float32)
        w = rng.standard_normal(8).astype(np.float32)
        np.testing.assert_allclose(qk_norm(x, w, 1e-6), oracle_rms(x, w, 1e-6), atol=1e-6)

    def test_fp16_activation_fp32_weight_promotes(self):
        x = np.array([[1.0, 3.0]], dtype=np.float16)
        w = np.array([1.0, 1.0], dtype=np.float32)
        y = qk_norm(x, w)
        self.assertEqual(y.dtype, np.float32)
        # inv_rms = 1/sqrt(mean(1, 9)+eps) = 1/sqrt(5)
        expected = np.array([[1.0, 3.0]], dtype=np.float32) / np.sqrt(np.float32(5.0))
        np.testing.assert_allclose(y, expected, atol=2e-3)

    def test_matched_fp16_stays_fp16(self):
        x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float16)
        w = np.ones(4, dtype=np.float16)
        y = qk_norm(x, w)
        self.assertEqual(y.dtype, np.float16)

    def test_rope_matches_split_formula_and_head_broadcast(self):
        rng = np.random.default_rng(11)
        q = rng.standard_normal((2, 3, 5, 8)).astype(np.float32)
        cos = rng.standard_normal((2, 3, 8)).astype(np.float32)
        sin = rng.standard_normal((2, 3, 8)).astype(np.float32)
        np.testing.assert_allclose(apply_rope(q, cos, sin), oracle_rope(q, cos, sin), atol=1e-6)

    def test_partial_rope_keeps_tail(self):
        x = np.arange(6, dtype=np.float32)
        cos = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        sin = np.zeros(4, dtype=np.float32)
        y = apply_rope(x, cos, sin)
        np.testing.assert_allclose(y[:4], x[:4], atol=1e-6)
        np.testing.assert_allclose(y[4:], x[4:], atol=1e-6)

    def test_fused_equals_norm_then_rope(self):
        rng = np.random.default_rng(3)
        q = rng.standard_normal((2, 4, 4, 8)).astype(np.float32)
        k = rng.standard_normal((2, 4, 2, 8)).astype(np.float32)
        wq = rng.standard_normal(8).astype(np.float32) * 0.1 + 1
        wk = rng.standard_normal(8).astype(np.float32) * 0.1 + 1
        cos = np.cos(rng.standard_normal((2, 4, 8))).astype(np.float32)
        sin = np.sin(rng.standard_normal((2, 4, 8))).astype(np.float32)
        q_f, k_f = qk_norm_rope(q, k, wq, wk, cos, sin, 1e-5)
        q_ref = apply_rope(qk_norm(q, wq, 1e-5), cos, sin)
        k_ref = apply_rope(qk_norm(k, wk, 1e-5), cos, sin)
        np.testing.assert_allclose(q_f, q_ref, atol=1e-6)
        np.testing.assert_allclose(k_f, k_ref, atol=1e-6)
        np.testing.assert_allclose(q_f, oracle_rope(oracle_rms(q, wq, 1e-5), cos, sin), atol=1e-6)

    def test_zero_vector_is_zero(self):
        x = np.zeros((1, 4), dtype=np.float32)
        w = np.array([2.0, -3.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(qk_norm(x, w), 0.0, atol=0)

    def test_rejects_odd_rotary(self):
        x = np.ones((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_rope(x, np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))

    def test_rejects_bad_weight(self):
        x = np.ones((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            qk_norm(x, np.ones(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
