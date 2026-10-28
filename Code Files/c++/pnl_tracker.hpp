// Cash, inventory, fees and funding accounting.

#pragma once

namespace hft {

struct PnLSnapshot {
    double cash{};        // realized cash incl. fees + funding
    double inventory{};   // base units, signed
    double fees{};        // cumulative fee cost (rebates negative)
    double funding{};     // cumulative discrete settlement P&L (signed)
    double mtm(double price) const noexcept {
        return cash + inventory * price;
    }
};

class PnLTracker {
public:
    PnLTracker(double maker_fee, double taker_fee) noexcept
        : maker_fee_(maker_fee), taker_fee_(taker_fee) {}

    // Maker fill at our own limit price. side: +1 we bought, -1 we sold.
    void apply_maker_fill(int side, double price, double qty) noexcept {
        const double notional = price * qty;
        const double fee = maker_fee_ * notional;
        s_.cash += (side > 0 ? -notional : notional);
        s_.cash -= fee;
        s_.fees += fee;
        s_.inventory += side * qty;
        ++n_fills_;
    }

    void apply_settlement(double mark_price, double funding_rate) noexcept {
        const double fee = s_.inventory * mark_price * funding_rate;
        s_.cash -= fee;
        s_.funding -= fee;
    }

    void liquidate(double bid0, double ask0) noexcept {
        if (s_.inventory > 1e-12 || s_.inventory < -1e-12) {
            const double px = s_.inventory > 0 ? bid0 : ask0;
            const double notional =
                (s_.inventory > 0 ? s_.inventory : -s_.inventory) * px;
            s_.cash += s_.inventory * px;
            const double fee = taker_fee_ * notional;
            s_.cash -= fee;
            s_.fees += fee;
            s_.inventory = 0.0;
            ++n_fills_;
        }
    }

    const PnLSnapshot& snap() const noexcept { return s_; }
    double inventory() const noexcept { return s_.inventory; }
    long n_fills() const noexcept { return n_fills_; }

private:
    double maker_fee_{};
    double taker_fee_{};
    PnLSnapshot s_{};
    long n_fills_{0};
};

} // namespace hft
