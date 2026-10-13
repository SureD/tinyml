# Hardware

Primary target: Apple Silicon MacBook Air. Unified memory, not a Max or Ultra. The repo does not record the Air's RAM, so the design carries two plan caps:

- 8 GB Air: process RSS under about 2 GB. Resident weights are int4 text.
- 16 GB Air: process RSS under about 4 GB. Bring-up may use BF16 weights; int8 is the comfortable resident format if int4 loses cosine.

`sysctl hw.memsize` picks the cap. Metal is the shipping backend. Compute on the Air is float32. Do not copy timings from `docs/metal_backend_validation.md` (that run was an M5 and TinyLlama).

Secondary target: NVIDIA GeForce RTX 3060. The user confirmed the name "JTX3060". Assume 12 GB VRAM. Note the 8 GB desktop variant and check with `nvidia-smi --query-gpu=name,memory.total --format=csv`. Text bfloat16, int8, and int4 all fit both sizes. Plan caps: 4 GB of the 12 GB card, 3 GB of the 8 GB card. CUDA, if added, is another backend for the same `embed()` call. Ampere may compute bfloat16. Float16 stays off. The 3060 does not set the phase order or the Air budgets.

Text only on both machines. Vision and audio towers stay unloaded.
