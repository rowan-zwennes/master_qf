// Reader and interpolator for the Regime II log-omega lookup table.
//
// Binary layout: 256-byte little-endian header, then n_q rows of float32
// log-omega, each row zero-padded to a 64-byte multiple. Row 0 is q = q_max
// (state order q = Q..-Q); column j is backward time u = j * du_ms.
// Quote depths are pre-computed at load, so the read path is two fused
// multiply-adds per side with no log or exp.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hft {

struct LutHeader {
    std::uint32_t n_q{}, n_u{}, du_ms{};
    std::int32_t q_min{}, q_max{};
    double gamma{}, k{}, A{}, rho{}, f_t{}, F_t{}, sigma{}, alpha_ml{},
        u_star_s{};
    std::uint32_t epoch_index{}, body_crc32{};
    std::uint64_t build_ts_ns{};
};

class RiccatiLUT {
public:
    RiccatiLUT() = default;
    ~RiccatiLUT();
    RiccatiLUT(const RiccatiLUT&) = delete;
    RiccatiLUT& operator=(const RiccatiLUT&) = delete;

    bool load(std::string_view path);
    bool reload_and_swap(std::string_view path);

    bool ready() const noexcept { return active_.load(std::memory_order_acquire) != nullptr; }
    const LutHeader& header() const noexcept;
    double u_star_seconds() const noexcept;
    const std::string& last_error() const noexcept { return err_; }

    struct D2 { double bid, ask; };
    D2 depths(int q, double u_seconds) const noexcept;

    bool has_sens() const noexcept;

    D2 depths_drift(int q, double u_seconds, double q0) const noexcept;

    // log-ω_q(u) (interpolated), for parity assertions, not the hot path.
    double log_omega(int q, double u_seconds) const noexcept;

private:
    struct Mapping;                          // owns the locked buffers
    static Mapping* build_mapping(std::string_view path, std::string& err);
    static void destroy_mapping(Mapping* m);

    std::atomic<const Mapping*> active_{nullptr};
    std::string err_;
};

} // namespace hft
