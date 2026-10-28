// The six quoting strategies of the ablation.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ema_drift.hpp"
#include "fill_simulator.hpp"
#include "glt_asymptotic.hpp"
#include "pnl_tracker.hpp"
#include "regime1_reader.hpp"
#include "replay_reader.hpp"
#include "reservation_price.hpp"
#include "riccati_lut.hpp"

namespace hft {

struct EngineConfig {
    double gamma{2.0e-5};
    int Q{10};
    // MUST mirror run_simulation.SimConfig (VIP0; positive = cost, not rebate)
    double maker_fee{2.0e-4};
    double taker_fee{5.0e-4};
    double delta_min{0.0};
    double fixed_half_spread{5.0};
    double ema_half_life_s{10.0};
    double post_gap_warmup_s{60.0};
    enum class MlMode { Horizon, Defensive };
    MlMode ml_mode{MlMode::Horizon};
    double ml_signal_horizon_s{10.0};   // h for both modes
};

struct StrategyResult {
    int sid{};
    double terminal_pnl{};
    double fees{};
    double funding{};
    long n_fills{};
    double mean_abs_inv{};
    double frac_time_at_cap{};
    double downtime_s{};
};

struct FillRecord {
    int sid{};
    std::int64_t ts_ms{};
    int side{};            // +1 we bought (bid fill), -1 we sold (ask fill)
    double price{};
    double qty{};
    double fee{};          // signed; maker rebate negative, taker liq positive
    double inv_after{};    // inventory immediately after this fill
    bool swept{};          // True if via sweep-through (vs at-level); mirrors
};

struct LutTimeline {
    std::vector<std::int64_t> ts;        // activation ts (ms), chronological
    std::vector<std::string> paths;      // HFTL file paths, parallel to ts
    std::size_t size() const noexcept { return ts.size(); }
};

class ReplayEngine {
public:
    ReplayEngine(const ReplayDay& day, const EngineConfig& cfg,
                 const RiccatiLUT* lut = nullptr,
                 const LutTimeline* timeline = nullptr,
                 const Regime1Reader* r1_reader = nullptr) noexcept
        : day_(day), cfg_(cfg), lut_(lut), timeline_(timeline),
          r1_reader_(r1_reader) {}

    std::vector<StrategyResult> run(const std::vector<int>& sids,
                                    std::vector<std::uint64_t>* latency_ns
                                    = nullptr,
                                    std::vector<FillRecord>* fills_out
                                    = nullptr);

private:
    const ReplayDay& day_;
    EngineConfig cfg_;
    const RiccatiLUT* lut_;
    const LutTimeline* timeline_;
    const Regime1Reader* r1_reader_;
};

} // namespace hft
