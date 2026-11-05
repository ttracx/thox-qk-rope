# Release gate

This tree is a source release. It is not a Kernel Hub artifact.

1. `python3 -m unittest tests.test_reference tests.test_spec -v` passes on the machine that cuts the tag.
2. `nix flake update` produces a lockfile that is committed before upload. This repository does not vendor a lock.
3. `kernel-builder build-and-copy` succeeds for the CPU and CUDA variants in `build.toml`.
4. A device parity run compares the built ops to `torch-ext/thox_qk_rope/reference.py` on float32 for at least: head dims 64 and 128, rotary equal to the head dim and rotary 64 with a tail, batch-seq-heads layout, and one decode-shaped seq of 1. Fail closed on mismatch.
5. Only after that run may anyone upload with `kernel-builder upload` to `Thox-ai/thox-qk-rope` on the kernel hub.
6. Half precision, bfloat16, and Metal stay out of the card until the same parity run exists for them. The Metal file in this tree is not on that path.
7. No tokens-per-second, joules, or speedup number is added to the card from a single uncommitted microbenchmark.

`thox-edge-ops` stays on its own repository and its own numerical contract.
