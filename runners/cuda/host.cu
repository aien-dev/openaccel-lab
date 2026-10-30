// SPDX-License-Identifier: Apache-2.0
// OpenAccel Lab CUDA entrant host. The runner writes workloads/<name>/kernel.cu followed by this file
// into one translation unit; kernel.cu defines
//     void oal_launch(const long long *p, void *const *buf);
// p holds the case parameters in manifest order, buf the device buffers: inputs, outputs, then the
// scratch buffers whose byte sizes are this program's arguments.
// stdin: one case container (docs/CONTAINER.md). stdout: outputs, each padded to 8 bytes.
// stderr: "oal-time-ns N" per timed repeat (launch + cudaDeviceSynchronize, host clock).
// Exit 2 on a malformed case, 3 on a CUDA error.
// The HIP entrant is this file and kernel.cu passed through hipify-perl unchanged by hand.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define OAL_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(3); } } while (0)

static unsigned long long oal_rd64(const unsigned char *b) {
    unsigned long long v = 0;
    for (int k = 7; k >= 0; k--) v = (v << 8) | b[k];
    return v;
}

static size_t oal_pad8(size_t n) { return (n + 7) / 8 * 8; }

int main(int argc, char **argv) {
    std::vector<unsigned char> in;
    unsigned char chunk[65536];
    size_t got;
    while ((got = std::fread(chunk, 1, sizeof chunk, stdin)) > 0) in.insert(in.end(), chunk, chunk + got);
    if (in.size() < 48 || std::memcmp(in.data(), "OALCASE1", 8) != 0) { std::fprintf(stderr, "bad case\n"); return 2; }
    unsigned long long np = oal_rd64(&in[8]), nin = oal_rd64(&in[16]), nout = oal_rd64(&in[24]);
    unsigned long long warm = oal_rd64(&in[32]), reps = oal_rd64(&in[40]);
    if (np > 32 || nin + nout > 32) { std::fprintf(stderr, "bad case counts\n"); return 2; }
    size_t o = 48;
    std::vector<long long> p(np + 1);
    for (unsigned long long k = 0; k < np; k++, o += 8) p[k] = (long long)oal_rd64(&in[o]);
    std::vector<size_t> sz(nin + nout);
    for (unsigned long long k = 0; k < nin + nout; k++, o += 8) sz[k] = (size_t)oal_rd64(&in[o]);
    int nscr = argc - 1;
    std::vector<void *> buf(nin + nout + nscr + 1);
    for (unsigned long long k = 0; k < nin; k++) {
        if (o + sz[k] > in.size()) { std::fprintf(stderr, "case truncated\n"); return 2; }
        OAL_CK(cudaMalloc(&buf[k], sz[k] ? sz[k] : 1));
        OAL_CK(cudaMemcpy(buf[k], &in[o], sz[k], cudaMemcpyHostToDevice));
        o += oal_pad8(sz[k]);
    }
    for (unsigned long long k = nin; k < nin + nout; k++) {
        OAL_CK(cudaMalloc(&buf[k], sz[k] ? sz[k] : 1));
        OAL_CK(cudaMemset(buf[k], 0xFF, sz[k]));
    }
    for (int s = 0; s < nscr; s++) {
        size_t n = (size_t)std::strtoull(argv[1 + s], nullptr, 10);
        OAL_CK(cudaMalloc(&buf[nin + nout + s], n ? n : 1));
        OAL_CK(cudaMemset(buf[nin + nout + s], 0, n));
    }
    OAL_CK(cudaDeviceSynchronize());
    for (unsigned long long r = 0; r < warm + reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        oal_launch(p.data(), buf.data());
        OAL_CK(cudaGetLastError());
        OAL_CK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        if (r >= warm)
            std::fprintf(stderr, "oal-time-ns %lld\n",
                         (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    std::vector<unsigned char> out;
    for (unsigned long long k = nin; k < nin + nout; k++) {
        size_t base = out.size();
        out.resize(base + oal_pad8(sz[k]), 0);
        OAL_CK(cudaMemcpy(&out[base], buf[k], sz[k], cudaMemcpyDeviceToHost));
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    return 0;
}
