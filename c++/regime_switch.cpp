// Regime I / Regime II switch.

#include "regime_switch.hpp"

namespace hft {

Regime pick_regime(double u_seconds_to_funding,
                   bool funding_transition,
                   double u_star_seconds) {
    if (funding_transition) return Regime::Transition;
    return (u_seconds_to_funding <= u_star_seconds) ? Regime::RiccatiLUT
                                                    : Regime::Asymptotic;
}

} // namespace hft
