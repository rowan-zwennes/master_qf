// Last-in-queue position tracking.

#include "queue_position.hpp"

#include <algorithm>
#include <cmath>

namespace hft {

namespace {

// Where does a trade printing at price `tp` sit relative to our order?
enum class Reach { NotReached, AtLevel, SweptThrough };

Reach classify(const PassiveOrder& o, double tp, double price_eps) {
    const double diff = tp - o.price;
    if (std::fabs(diff) <= price_eps) {
        return Reach::AtLevel;
    }
    if (o.side > 0) {
        return (tp > o.price) ? Reach::NotReached : Reach::SweptThrough;
    }
    return (tp < o.price) ? Reach::NotReached : Reach::SweptThrough;
}

} // namespace

int QueuePositionTracker::place(int side, double price, double our_qty,
                                double queue_ahead, std::uint64_t ts_ms) {
    PassiveOrder o;
    o.side = side;
    o.price = price;
    o.our_qty = our_qty;
    o.queue_ahead = queue_ahead;
    o.placed_event_time_ms = ts_ms;
    o.id = next_id_++;
    orders_.push_back(o);
    return o.id;
}

std::vector<FillEvent> QueuePositionTracker::on_trade(double price, double qty,
                                                      int taker_side,
                                                      std::uint64_t ts_ms) {
    std::vector<FillEvent> fills;
    if (qty <= qty_eps_ || taker_side == 0) {
        return fills;
    }

    const int target_side = -taker_side;

    std::vector<std::size_t> idx;
    idx.reserve(orders_.size());
    for (std::size_t i = 0; i < orders_.size(); ++i) {
        if (orders_[i].side == target_side) idx.push_back(i);
    }
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        const PassiveOrder& oa = orders_[a];
        const PassiveOrder& ob = orders_[b];
        if (oa.price != ob.price) {
            // higher priority first: bids want larger price, asks smaller price
            return (target_side > 0) ? (oa.price > ob.price) : (oa.price < ob.price);
        }
        return oa.id < ob.id;  // FIFO among equal prices
    });

    double remaining = qty;  // trade volume budget shared across our same-side orders
    for (std::size_t k = 0; k < idx.size(); ++k) {
        PassiveOrder& o = orders_[idx[k]];
        if (ts_ms < o.placed_event_time_ms + order_latency_ms_) {
            continue;  // still in flight: not resting when this trade hit
        }
        const Reach r = classify(o, price, price_eps_);
        if (r == Reach::NotReached) {
            continue;
        }
        if (r == Reach::SweptThrough) {
            o.queue_ahead = 0.0;
            const double f = std::min(o.our_qty, remaining);
            if (f > qty_eps_) {
                fills.push_back({o.price, f, o.side, ts_ms,
                                 o.placed_event_time_ms, /*swept=*/true});
                o.our_qty -= f;
                remaining -= f;
            }
            continue;
        }
        const double d = std::min(o.queue_ahead, remaining);
        o.queue_ahead -= d;
        remaining -= d;
        if (remaining > qty_eps_ && o.queue_ahead <= qty_eps_) {
            const double f = std::min(o.our_qty, remaining);
            if (f > qty_eps_) {
                fills.push_back({o.price, f, o.side, ts_ms,
                                 o.placed_event_time_ms, /*swept=*/false});
                o.our_qty -= f;
                remaining -= f;
            }
        }
    }

    // Drop fully-filled orders so later trades cannot refill them.
    orders_.erase(std::remove_if(orders_.begin(), orders_.end(),
                                 [&](const PassiveOrder& o) {
                                     return o.our_qty <= qty_eps_;
                                 }),
                  orders_.end());
    return fills;
}

void QueuePositionTracker::cancel(int id) {
    orders_.erase(std::remove_if(orders_.begin(), orders_.end(),
                                 [&](const PassiveOrder& o) { return o.id == id; }),
                  orders_.end());
}

void QueuePositionTracker::cancel_all() { orders_.clear(); }

const PassiveOrder* QueuePositionTracker::find(int id) const {
    for (const auto& o : orders_) {
        if (o.id == id) return &o;
    }
    return nullptr;
}

} // namespace hft
