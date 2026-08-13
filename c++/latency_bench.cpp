// Timing harness for the quoting hot path.

// Latency micro-benchmarks + HFTB dump. See latency_bench.hpp.

#include "latency_bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

#include "glt_asymptotic.hpp"
#include "regime1_reader.hpp"
#include "riccati_lut.hpp"

namespace hft {

namespace {

using clk = std::chrono::steady_clock;

double batch_ns(clk::time_point a, clk::time_point b, std::size_t iters) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                   .count())
           / static_cast<double>(iters);
}

} // namespace

std::vector<double> bench_glt_closed_form(std::size_t batches,
                                          std::size_t batch_size) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    // pre-generate inputs so the loop measures the formula, not the RNG
    std::vector<int> qv(batch_size);
    std::vector<double> av(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        qv[i] = static_cast<int>(u01(rng) * 21.0) - 10;
        av[i] = (u01(rng) - 0.5) * 0.4;
    }
    const glt::MarketConsts mc(2.0e-5, 2.5, 20.0, 0.145);
    std::vector<double> out;
    out.reserve(batches);
    double sink = 0.0;
    for (std::size_t b = 0; b < batches; ++b) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < batch_size; ++i) {
            const auto d = glt::depths(mc, qv[i], av[i]);
            sink += d.bid - d.ask;
        }
        const auto t1 = clk::now();
        out.push_back(batch_ns(t0, t1, batch_size));
    }
    // sink escapes through a volatile so the loop cannot be folded away
    volatile double guard = sink;
    (void)guard;
    return out;
}

std::vector<double> bench_lut_interp(const RiccatiLUT& lut,
                                     std::size_t batches,
                                     std::size_t batch_size) {
    std::mt19937_64 rng(43);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int Q = lut.header().q_max;
    const double u_max =
        (lut.header().n_u - 1) * (lut.header().du_ms / 1000.0);
    std::vector<int> qv(batch_size);
    std::vector<double> uv(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        qv[i] = static_cast<int>(u01(rng) * (2 * Q + 1)) - Q;
        uv[i] = u01(rng) * u_max;
    }
    std::vector<double> out;
    out.reserve(batches);
    double sink = 0.0;
    for (std::size_t b = 0; b < batches; ++b) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < batch_size; ++i) {
            const auto d = lut.depths(qv[i], uv[i]);
            sink += d.bid + d.ask;
        }
        const auto t1 = clk::now();
        out.push_back(batch_ns(t0, t1, batch_size));
    }
    volatile double guard = sink;
    (void)guard;
    return out;
}

std::vector<double> bench_regime1_read(const Regime1Reader& reader,
                                       std::size_t batches,
                                       std::size_t batch_size) {
    using S = Regime1Reader::Stream;
    std::mt19937_64 rng(44);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int Q = reader.Q();
    const std::int64_t lo = reader.ts_first(S::Auto);
    const std::int64_t hi = reader.ts_last(S::Auto);
    const double span = static_cast<double>(hi - lo);
    std::vector<int> qv(batch_size);
    std::vector<std::int64_t> tv(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        qv[i] = static_cast<int>(u01(rng) * (2 * Q + 1)) - Q;
        tv[i] = lo + static_cast<std::int64_t>(u01(rng) * span);
    }
    std::sort(tv.begin(), tv.end());
    // production calibration (the mc the live engine recomputes depths with)
    const glt::MarketConsts mc(2.0e-5, 4.575, 0.2742, 0.0900);
    std::vector<double> out;
    out.reserve(batches);
    double sink = 0.0;
    for (std::size_t b = 0; b < batches; ++b) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < batch_size; ++i) {
            const auto d = reader.depths(S::Auto, tv[i], qv[i], mc);
            sink += d.bid + d.ask;
        }
        const auto t1 = clk::now();
        out.push_back(batch_ns(t0, t1, batch_size));
    }
    volatile double guard = sink;
    (void)guard;
    return out;
}

bool latency_dump(const char* path, const std::vector<StageSamples>& stages) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const char magic[4] = {'H', 'F', 'T', 'B'};
    const std::uint32_t version = 1;
    const auto n_stages = static_cast<std::uint32_t>(stages.size());
    const std::uint32_t pad = 0;
    std::fwrite(magic, 1, 4, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&n_stages, 4, 1, f);
    std::fwrite(&pad, 4, 1, f);
    bool ok = true;
    for (const auto& st : stages) {
        char name[24] = {};
        std::strncpy(name, st.name.c_str(), sizeof(name) - 1);
        std::fwrite(name, 1, sizeof(name), f);
        std::vector<std::uint64_t> samples;
        samples.reserve(st.ns.size() + st.ns_per_op.size());
        for (std::uint64_t v : st.ns) samples.push_back(v * 1000ull);
        for (double v : st.ns_per_op)
            samples.push_back(static_cast<std::uint64_t>(v * 1000.0));
        const std::uint64_t n = samples.size();
        std::fwrite(&n, 8, 1, f);
        if (n && std::fwrite(samples.data(), 8, n, f) != n) ok = false;
    }
    std::fclose(f);
    return ok;
}

} // namespace hft
