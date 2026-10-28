// Lock-free parameter block.

#include "params_atomic.hpp"

namespace hft {

ParamsAtomic::ParamsAtomic() {
    active_.store(&buffers_[0], std::memory_order_release);
}

void ParamsAtomic::publish(const ParamSnapshot& src) noexcept {
    const ParamSnapshot* cur = active_.load(std::memory_order_acquire);
    ParamSnapshot* inactive = (cur == &buffers_[0]) ? &buffers_[1]
                                                    : &buffers_[0];
    *inactive = src;
    inactive->seq = cur->seq + 1;
    active_.store(inactive, std::memory_order_release);
}

ParamSnapshot ParamsAtomic::snapshot() const noexcept {
    return *active_.load(std::memory_order_acquire);
}

} // namespace hft
