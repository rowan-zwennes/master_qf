// Last-in-queue position tracking for a resting order.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hft {

// A resting maker order under the last-in-queue model.
struct PassiveOrder {
    int side{};                          // +1 bid (we buy), -1 ask (we sell)
    double price{};                      // our limit price (USDT)
    double our_qty{};                    // our remaining size (base units)
    double queue_ahead{};                // displayed size still ahead of us (ratchet)
    std::uint64_t placed_event_time_ms{};
    int id{-1};                          // stable handle returned by place()
};

// One execution of (part of) one of our resting orders.
struct FillEvent {
    double price{};                      // our limit price (maker fills at own price)
    double qty{};                        // filled size (base units)
    int side{};                          // +1 we bought, -1 we sold
    std::uint64_t ts_ms{};               // event time of the triggering trade
    std::uint64_t placed_ts_ms{};        // when the filled order was posted
    bool swept{};                        // True if via the sweep-through path
                                         // (vs at-level); mirrors fill_core_reference
};

class QueuePositionTracker {
public:
    // Place a resting maker order; returns its stable id.
    int place(int side, double price, double our_qty, double queue_ahead,
              std::uint64_t ts_ms);

    std::vector<FillEvent> on_trade(double price, double qty, int taker_side,
                                    std::uint64_t ts_ms);

    void cancel(int id);
    void cancel_all();

    std::size_t active_count() const { return orders_.size(); }
    // Read-only handle for inspection / tests; nullptr if not found.
    const PassiveOrder* find(int id) const;

    void set_price_eps(double eps) { price_eps_ = eps; }
    void set_qty_eps(double eps) { qty_eps_ = eps; }

    void set_order_latency_ms(std::uint64_t l) { order_latency_ms_ = l; }

private:
    std::vector<PassiveOrder> orders_;
    int next_id_{0};
    double price_eps_{1e-9};
    double qty_eps_{1e-12};
    std::uint64_t order_latency_ms_{0};
};

} // namespace hft
