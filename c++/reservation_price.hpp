// Reservation price and quote assembly.

#pragma once

#include <cmath>

#include "glt_asymptotic.hpp"
#include "riccati_lut.hpp"

namespace hft {

struct QuoteDepths { double bid, ask; };

struct StrategyFlags {
    bool naive{false};
    bool funding{false};   // consult the LUT inside u*
    bool ml_shift{false};  // superpose the α_ML shift on LUT depths (s6)
};

inline QuoteDepths quote_depths(const StrategyFlags& fl,
                                const glt::MarketConsts& mc, int q,
                                double alpha, double u_s,
                                const RiccatiLUT* lut,
                                double fixed_half_spread,
                                const QuoteDepths* regime1 = nullptr,
                                bool force_r2 = false) noexcept {
    if (fl.naive) return {fixed_half_spread, fixed_half_spread};
    if (fl.funding && lut != nullptr && lut->ready()
        && (force_r2 || u_s <= lut->u_star_seconds())) {
        if (fl.ml_shift && lut->has_sens()) {
            const auto d = lut->depths_drift(q, u_s, alpha * mc.inv_gs2);
            return {d.bid, d.ask};
        }
        auto d = lut->depths(q, u_s);
        if (fl.ml_shift) {
            // Legacy additive fallback: LUT with no sensitivity companion.
            const double shift = glt::ml_quote_shift(mc, alpha);
            d.bid -= shift;
            d.ask += shift;
        }
        return {d.bid, d.ask};
    }
    if (regime1 != nullptr) return *regime1;        // exact f^0 (drift baked)
    const auto d = glt::depths(mc, q, alpha);        // Gaussian fallback
    return {d.bid, d.ask};
}

} // namespace hft
