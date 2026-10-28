// Regime I / Regime II switch on backward time u.

#pragma once

namespace hft {

enum class Regime { Asymptotic, RiccatiLUT, Transition };

Regime pick_regime(double u_seconds_to_funding,
                   bool   funding_transition,
                   double u_star_seconds);

} // namespace hft
