// Lock-free parameter block, swapped by atomic pointer.

#pragma once

#include <atomic>
#include <cstdint>

namespace hft {

struct alignas(64) ParamSnapshot {
    double alpha_ml{};
    double sigma{};
    double A{};
    double k{};
    double rho{};
    double f_t{};
    double gamma{};
    double eta{};
    std::uint64_t seq{};
};

class ParamsAtomic {
public:
    ParamsAtomic();
    void publish(const ParamSnapshot& src) noexcept;
    ParamSnapshot snapshot() const noexcept;

private:
    alignas(64) ParamSnapshot buffers_[2]{};
    std::atomic<const ParamSnapshot*> active_{nullptr};
};

} // namespace hft
