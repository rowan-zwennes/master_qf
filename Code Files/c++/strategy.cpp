// The six quoting strategies of the ablation.

#include "strategy.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace hft {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Python: _round_down / _round_up (tick rounding with float-noise guard)
inline double round_down(double x, double tick) noexcept {
    return std::floor(x / tick + 1e-9) * tick;
}
inline double round_up(double x, double tick) noexcept {
    return std::ceil(x / tick - 1e-9) * tick;
}

inline double queue_ahead(double price, const double* prices,
                          const float* sizes, std::size_t L, int side,
                          double tick) noexcept {
    for (std::size_t i = 0; i < L; ++i) {
        if (std::fabs(prices[i] - price) < tick * 0.5)
            return static_cast<double>(sizes[i]);
    }
    const double deepest = prices[L - 1];
    if (((side > 0 && price < deepest) || (side < 0 && price > deepest))
        && deepest > 0.0)
        return static_cast<double>(sizes[L - 1]);
    return 0.0;
}

struct StratState {
    int sid{};
    StrategyFlags flags{};
    bool use_ar{false}, use_ml{false};
    EmaDrift ema;                          // RETIRED, never updated
    FillSimulator fills;
    PnLTracker pnl;
    int bid_id{-1}, ask_id{-1};
    double bid_px{kNaN}, ask_px{kNaN};
    double absq_sum{0.0};
    long cap_ticks{0};
    bool r2_latched{false};

    StratState(int s, const EngineConfig& cfg)
        : sid(s), ema(cfg.ema_half_life_s),
          pnl(cfg.maker_fee, cfg.taker_fee) {
        switch (s) {                       // sim_quote_engine.STRATEGIES
            case 1: flags.naive = true; break;
            case 2: break;
            case 3: use_ar = true; break;
            case 4: use_ml = true; break;
            case 5: flags.funding = true; break;
            case 6: flags.funding = true; flags.ml_shift = true;
                    use_ml = true; break;
            default: break;
        }
    }

    void cancel_all() {
        fills.cancel_all();
        bid_id = ask_id = -1;
        bid_px = ask_px = kNaN;
    }

    int q_lots(double qs, int Q) const noexcept {
        const double q = pnl.inventory() / qs;
        int qi = static_cast<int>(std::nearbyint(q));
        if (qi > Q) qi = Q;
        if (qi < -Q) qi = -Q;
        return qi;
    }
};

} // namespace

std::vector<StrategyResult> ReplayEngine::run(
    const std::vector<int>& sids, std::vector<std::uint64_t>* latency_ns,
    std::vector<FillRecord>* fills_out) {
    const ReplayDay& d = day_;
    const auto n = static_cast<std::size_t>(d.n_ticks);
    const auto L = static_cast<std::size_t>(d.n_levels);
    const double tick = d.tick;
    const double qs = d.quote_size;
    const int Q = cfg_.Q;

    std::vector<StratState> states;
    states.reserve(sids.size());
    for (int s : sids) {
        states.emplace_back(s, cfg_);
        // Python: tracker.price_eps = cfg.tick * 0.499
        states.back().fills.tracker().set_price_eps(tick * 0.499);
    }

    std::size_t trade_cursor = 0, settle_i = 0, pause_i = 0;
    bool paused_prev = false;
    std::int64_t downtime_ms = 0;
    std::int64_t live_ticks = 0;   // Python live_ticks twin (diagnostics denom)
    const std::int64_t warmup_ms =
        static_cast<std::int64_t>(cfg_.post_gap_warmup_s * 1000.0);
    std::int64_t warmup_until = -1;

    RiccatiLUT swap_lut;
    const RiccatiLUT* active_lut = lut_;
    std::size_t lut_cursor = 0;
    glt::MarketConsts mc;
    double mc_sigma = -1.0, mc_A = -1.0, mc_k = -1.0;

    auto apply_settlement = [&](std::size_t idx) {
        for (auto& st : states) st.r2_latched = false;
        const double mark = d.settle_mark[idx];
        const double fr = d.settle_rate[idx];
        if (std::isnan(mark)) return;
        for (auto& st : states) st.pnl.apply_settlement(mark, fr);
    };

    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t t = d.tick_ts[i];

        while (trade_cursor < d.n_trades && d.trade_ts[trade_cursor] < t) {
            const std::int64_t tt = d.trade_ts[trade_cursor];
            while (settle_i < d.n_settle && d.settle_ts[settle_i] <= tt) {
                apply_settlement(settle_i);
                ++settle_i;
            }
            while (pause_i < d.n_pause && tt >= d.pause[2 * pause_i + 1])
                ++pause_i;
            if (pause_i < d.n_pause && d.pause[2 * pause_i] <= tt) {
                for (auto& st : states)
                    if (st.fills.active_count()) st.cancel_all();
                ++trade_cursor;
                continue;
            }
            const double px = d.trade_px[trade_cursor];
            const double qty = d.trade_qty[trade_cursor];
            const int side = d.trade_side[trade_cursor];
            for (auto& st : states) {
                if (st.fills.active_count() == 0) continue;
                for (const FillEvent& f : st.fills.on_trade(
                         px, qty, side, static_cast<std::uint64_t>(tt))) {
                    st.pnl.apply_maker_fill(f.side, f.price, f.qty);
                    // fee + inv-after exactly as run_simulation records them
                    if (fills_out) {
                        const double fee = cfg_.maker_fee * f.price * f.qty;
                        fills_out->push_back({st.sid, tt, f.side, f.price,
                                              f.qty, fee, st.pnl.inventory(),
                                              f.swept});
                    }
                    if (f.side > 0) {
                        if (std::fabs(f.price - st.bid_px) < tick * 0.5
                            && st.fills.find(st.bid_id) == nullptr) {
                            st.bid_id = -1;
                            st.bid_px = kNaN;
                        }
                    } else {
                        if (std::fabs(f.price - st.ask_px) < tick * 0.5
                            && st.fills.find(st.ask_id) == nullptr) {
                            st.ask_id = -1;
                            st.ask_px = kNaN;
                        }
                    }
                }
            }
            ++trade_cursor;
        }
        while (settle_i < d.n_settle && d.settle_ts[settle_i] <= t) {
            apply_settlement(settle_i);
            ++settle_i;
        }

        while (pause_i < d.n_pause && t >= d.pause[2 * pause_i + 1]) ++pause_i;
        const bool in_gap =
            (pause_i < d.n_pause && d.pause[2 * pause_i] <= t)
            || d.valid[i] == 0;
        if (in_gap) {
            if (!paused_prev)
                for (auto& st : states) st.cancel_all();
            paused_prev = true;
            downtime_ms += 100;
            continue;
        }
        if (paused_prev) {
            paused_prev = false;
            // post-gap warm-up: stay flat until sigma + book features refill
            warmup_until = t + warmup_ms;
        }

        const double mid = d.mid[i];
        const double bid0 = d.bid_p[i * L];
        const double ask0 = d.ask_p[i * L];
        if (!(bid0 > 0.0 && ask0 > bid0)) continue;

        ++live_ticks;
        if (d.sigma[i] != mc_sigma || d.A[i] != mc_A || d.k[i] != mc_k) {
            mc = glt::MarketConsts(cfg_.gamma, d.sigma[i], d.A[i], d.k[i]);
            mc_sigma = d.sigma[i];
            mc_A = d.A[i];
            mc_k = d.k[i];
        }

        if (timeline_) {
            while (lut_cursor < timeline_->size()
                   && timeline_->ts[lut_cursor] <= t) {
                if (!swap_lut.reload_and_swap(timeline_->paths[lut_cursor])) {
                    std::fprintf(stderr, "lut swap failed: %s\n",
                                 swap_lut.last_error().c_str());
                }
                active_lut = &swap_lut;
                ++lut_cursor;
            }
        }

        // quoting, 100 ms grid
        const double u = d.u_s[i];
        const bool in_warmup = t < warmup_until;
        for (auto& st : states) {
            const auto t_q0 = std::chrono::steady_clock::now();
            if (in_warmup) {
                if (st.bid_id >= 0 || st.ask_id >= 0) st.cancel_all();
                st.absq_sum += std::fabs(st.pnl.inventory());
                if (st.q_lots(qs, Q) >= Q || st.q_lots(qs, Q) <= -Q)
                    ++st.cap_ticks;
                continue;
            }
            double alpha = st.use_ml ? d.alpha[i]
                                : (st.use_ar ? d.alpha_ar[i] : 0.0);
            if (alpha != 0.0) {
                double shift_usd = alpha * cfg_.ml_signal_horizon_s;
                if (shift_usd > mc.c1) shift_usd = mc.c1;
                if (shift_usd < -mc.c1) shift_usd = -mc.c1;
                const double hdr = (r1_reader_ != nullptr)
                    ? r1_reader_->hdrift(t)
                    : kNaN;
                alpha = shift_usd / (std::isfinite(hdr) && hdr > 0.0
                                     ? hdr : (mc.inv_gs2 * mc.scale));
            }
            const int q_int = st.q_lots(qs, Q);
            QuoteDepths r1buf, r1autobuf;
            const QuoteDepths* r1 = nullptr;       // stream for the MAIN quote
            const QuoteDepths* r1_auto = nullptr;  // autonomous (defensive base)
            if (r1_reader_ != nullptr) {
                if (r1_reader_->ready(Regime1Reader::Stream::Auto)) {
                    const auto d = r1_reader_->depths(
                        Regime1Reader::Stream::Auto, t, q_int, mc);
                    r1autobuf = {d.bid, d.ask};
                    r1_auto = &r1autobuf;
                }
                if (alpha != 0.0) {
                    const auto sm =
                        Regime1Reader::stream_for(st.use_ml, st.use_ar);
                    if (r1_reader_->ready(sm)) {
                        const auto d = r1_reader_->depths(sm, t, q_int, mc);
                        r1buf = {d.bid, d.ask};
                        r1 = &r1buf;
                    }
                } else {
                    r1 = r1_auto;   // alpha == 0 -> autonomous f^0
                }
            }
            if (st.flags.funding && active_lut != nullptr && active_lut->ready()
                && u <= active_lut->u_star_seconds())
                st.r2_latched = true;
            const bool force_r2 = st.r2_latched;
            QuoteDepths qd = quote_depths(
                st.flags, mc, q_int, alpha, u, active_lut,
                cfg_.fixed_half_spread, r1, force_r2);
            if (alpha != 0.0 && cfg_.ml_mode == EngineConfig::MlMode::Defensive) {
                const QuoteDepths qd0 = quote_depths(
                    st.flags, mc, q_int, 0.0, u, active_lut,
                    cfg_.fixed_half_spread, r1_auto, force_r2);
                if (qd0.bid > qd.bid) qd.bid = qd0.bid;
                if (qd0.ask > qd.ask) qd.ask = qd0.ask;
            }

            // Python: want_bid/want_ask test REAL inventory headroom
            const double inv = st.pnl.inventory();
            bool want_bid = (inv + qs <= Q * qs + 1e-9)
                            && std::isfinite(qd.bid);
            bool want_ask = (inv - qs >= -Q * qs - 1e-9)
                            && std::isfinite(qd.ask);
            double bpx = kNaN, apx = kNaN;
            if (want_bid) {
                const double db = qd.bid > cfg_.delta_min ? qd.bid
                                                          : cfg_.delta_min;
                bpx = round_down(mid - db, tick);
                bpx = round_down(bpx < ask0 - tick ? bpx : ask0 - tick, tick);
                if (bpx <= 0.0) want_bid = false;
            }
            if (want_ask) {
                const double da = qd.ask > cfg_.delta_min ? qd.ask
                                                          : cfg_.delta_min;
                apx = round_up(mid + da, tick);
                apx = round_up(apx > bid0 + tick ? apx : bid0 + tick, tick);
            }
            if (latency_ns) {
                latency_ns->push_back(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t_q0).count()));
            }

            // bid: keep resting order (and its queue position) if unchanged
            if (st.bid_id >= 0
                && (!want_bid || std::fabs(bpx - st.bid_px) > tick * 0.5)) {
                st.fills.cancel(st.bid_id);
                st.bid_id = -1;
                st.bid_px = kNaN;
            }
            if (want_bid && st.bid_id < 0) {
                const double qa = queue_ahead(bpx, &d.bid_p[i * L],
                                              &d.bid_q[i * L], L, +1, tick);
                st.bid_id = st.fills.place(+1, bpx, qs, qa,
                                           static_cast<std::uint64_t>(t));
                st.bid_px = bpx;
            }
            if (st.ask_id >= 0
                && (!want_ask || std::fabs(apx - st.ask_px) > tick * 0.5)) {
                st.fills.cancel(st.ask_id);
                st.ask_id = -1;
                st.ask_px = kNaN;
            }
            if (want_ask && st.ask_id < 0) {
                const double qa = queue_ahead(apx, &d.ask_p[i * L],
                                              &d.ask_q[i * L], L, -1, tick);
                st.ask_id = st.fills.place(-1, apx, qs, qa,
                                           static_cast<std::uint64_t>(t));
                st.ask_px = apx;
            }

            st.absq_sum += std::fabs(st.pnl.inventory());
            if (q_int >= Q || q_int <= -Q) ++st.cap_ticks;
        }
    }

    std::size_t last = n - 1;
    // Python: last valid tick with bid>0 AND ask>bid (not crossed/locked).
    while (last > 0 && !(d.bid_p[last * L] > 0.0
                         && d.ask_p[last * L] > d.bid_p[last * L]
                         && d.valid[last] != 0))
        --last;
    const double lb = d.bid_p[last * L];
    const double la = d.ask_p[last * L];
    const std::int64_t t_end = d.tick_ts[last];

    std::vector<StrategyResult> out;
    out.reserve(states.size());
    for (auto& st : states) {
        st.cancel_all();
        const double inv_before = st.pnl.inventory();
        if (lb > 0.0 && la > lb) st.pnl.liquidate(lb, la);
        if (fills_out && lb > 0.0 && la > lb
            && (inv_before > 1e-12 || inv_before < -1e-12)) {
            // Python records the liquidation as a fill (taker fee, inv->0)
            const double px = inv_before > 0.0 ? lb : la;
            const double q = inv_before > 0.0 ? inv_before : -inv_before;
            const double fee = cfg_.taker_fee * q * px;
            fills_out->push_back({st.sid, t_end,
                                  inv_before > 0.0 ? -1 : 1, px, q, fee, 0.0,
                                  false});
        }
        const PnLSnapshot& s = st.pnl.snap();
        out.push_back({st.sid, s.cash, s.fees, s.funding, st.pnl.n_fills(),
                       st.absq_sum
                           / static_cast<double>(live_ticks > 0 ? live_ticks : 1),
                       static_cast<double>(st.cap_ticks)
                           / static_cast<double>(live_ticks > 0 ? live_ticks : 1),
                       static_cast<double>(downtime_ms) / 1000.0});
    }
    return out;
}

} // namespace hft
