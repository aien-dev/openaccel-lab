# Case container (`OALCASE1`)

One case is one byte stream. `bin/oal-gen` writes it to stdout; every entrant program reads it on
stdin and writes its outputs to stdout. Because every entrant is a separate process that only sees
these bytes, a crash in one entrant cannot corrupt another, and identical input is provable: the
container's sha256 is recorded in every result line and the gate requires it to agree across runners.

All integers are little-endian.

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `OALCASE1` |
| 8 | 8 | u64 `nparam`: declared parameters, then derived parameters, in manifest order |
| 16 | 8 | u64 `nin`: input tensors |
| 24 | 8 | u64 `nout`: output tensors |
| 32 | 8 | u64 `warmup`: untimed launches before timing |
| 40 | 8 | u64 `repeats`: timed launches |
| 48 | 8*nparam | i64 parameter values |
| ... | 8*(nin+nout) | u64 byte sizes of each input, then each output |
| ... | ... | input tensors in manifest order, each padded with zero bytes to a multiple of 8 |

Outputs: the entrant writes each output tensor in manifest order, each padded to a multiple of 8,
and nothing else on stdout. Output buffers start filled with `0xFF` bytes in every runner, so an
element the kernel never writes is a NaN (floats) or -1 (integers) and fails the check instead of
passing by luck on zeroed memory.

Timing: for each timed launch the entrant writes `oal-time-ns N` on stderr, where `N` is the
wall-clock nanoseconds of that launch including the wait for completion and excluding host-device
copies. It may write `oal-version TEXT` lines naming its runtime. Anything else on stderr is a
diagnostic and is kept in the result's `log` field only on failure.

Exit codes of entrant programs: 0 success, 2 malformed case, 3 runtime error (device, launch or
API failure), 100+N an Omega `trap N` (9 bad header, 10 output size mismatch, 11 trailing bytes).
