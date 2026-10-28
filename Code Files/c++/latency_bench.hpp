// Timing harness for the quoting hot path.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hft {

struct StageSamples {
    std::string name;                    // truncated to 23 chars on dump
    std::vector<std::uint64_t> ns;       // raw per-op samples (ns; replay)
    std::vector<double> ns_per_op;       // batched samples (micro bench)
};

bool latency_dump(const char* path, const std::vector<StageSamples>& stages);

// Micro benches; each returns batched ns/op samples (one per batch).
std::vector<double> bench_glt_closed_form(std::size_t batches,
                                          std::size_t batch_size);
std::vector<double> bench_lut_interp(const class RiccatiLUT& lut,
                                     std::size_t batches,
                                     std::size_t batch_size);
std::vector<double> bench_regime1_read(const class Regime1Reader& reader,
                                       std::size_t batches,
                                       std::size_t batch_size);

} // namespace hft
