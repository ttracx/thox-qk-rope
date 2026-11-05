"""Compile spec.c and check it against the NumPy contract. fp32 only."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

from thox_qk_rope.reference import apply_rope, qk_norm  # noqa: E402


def _load():
    src = ROOT / "qk_rope_cpu" / "spec.c"
    lib = Path(tempfile.mkdtemp()) / "libthoxspec.so"
    subprocess.check_call(
        ["gcc", "-std=c11", "-O2", "-fPIC", "-shared", "-o", str(lib), str(src), "-lm"],
    )
    so = ctypes.CDLL(str(lib))
    so.thox_qk_norm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
    ]
    so.thox_apply_rope_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    return so


class SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = _load()

    def test_norm(self):
        rng = np.random.default_rng(19)
        x = rng.standard_normal((6, 8)).astype(np.float32)
        w = (rng.standard_normal(8) * 0.2 + 1).astype(np.float32)
        y = np.empty_like(x)
        self.lib.thox_qk_norm_f32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[0],
            x.shape[1],
            ctypes.c_float(1e-6),
        )
        np.testing.assert_allclose(y, qk_norm(x, w), atol=1e-6)

    def test_rope_head_broadcast_and_tail(self):
        rng = np.random.default_rng(23)
        tokens, heads, dim, rotary = 3, 4, 6, 4
        x = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
        cos = np.cos(rng.standard_normal((tokens, rotary))).astype(np.float32)
        sin = np.sin(rng.standard_normal((tokens, rotary))).astype(np.float32)
        y = np.empty_like(x)
        self.lib.thox_apply_rope_f32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            tokens,
            heads,
            dim,
            rotary,
        )
        np.testing.assert_allclose(y, apply_rope(x, cos, sin), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
