"""Python reference twin of the C++ last-in-queue fill core."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class PassiveOrder:
    side: int                  # +1 bid (we buy), -1 ask (we sell)
    price: float
    our_qty: float             # remaining size
    queue_ahead: float         # displayed size still ahead of us (ratchet)
    placed_event_time_ms: int
    id: int


@dataclass
class FillEvent:
    price: float               # our limit price (maker fills at own price)
    qty: float
    side: int                  # +1 we bought, -1 we sold
    ts_ms: int
    placed_ts_ms: int = -1     # when the filled order was posted (diagnostics)
    swept: bool = False        # True if via the sweep-through path (vs at-level);
                               # lets the sim measure per-strategy toxic-sweep exposure


# Reach classification: where a trade at price `tp` sits vs our order.
_NOT_REACHED, _AT_LEVEL, _SWEPT_THROUGH = 0, 1, 2


def _classify(o: PassiveOrder, tp: float, price_eps: float) -> int:
    if abs(tp - o.price) <= price_eps:
        return _AT_LEVEL
    if o.side > 0:
        return _NOT_REACHED if tp > o.price else _SWEPT_THROUGH
    # our ASK: a buy below us is not reached; a buy above swept through.
    return _NOT_REACHED if tp < o.price else _SWEPT_THROUGH


class QueuePositionTracker:
    def __init__(self, price_eps: float = 1e-9, qty_eps: float = 1e-12,
                 order_latency_ms: int = 0) -> None:
        self._orders: list[PassiveOrder] = []
        self._next_id = 0
        self.price_eps = price_eps
        self.qty_eps = qty_eps
        self.order_latency_ms = order_latency_ms

    def place(self, side: int, price: float, our_qty: float,
              queue_ahead: float, ts_ms: int) -> int:
        oid = self._next_id
        self._next_id += 1
        self._orders.append(
            PassiveOrder(side, price, our_qty, queue_ahead, ts_ms, oid))
        return oid

    def on_trade(self, price: float, qty: float, taker_side: int,
                 ts_ms: int) -> list[FillEvent]:
        fills: list[FillEvent] = []
        if qty <= self.qty_eps or taker_side == 0:
            return fills

        target_side = -taker_side  # sell(-1) hits bids(+1); buy(+1) hits asks(-1)

        # our orders on the hit side, in price-priority then FIFO order
        hit = [o for o in self._orders if o.side == target_side]
        hit.sort(key=lambda o: (-o.price if target_side > 0 else o.price, o.id))

        remaining = qty  # shared trade-volume budget across same-side orders
        for o in hit:
            if ts_ms < o.placed_event_time_ms + self.order_latency_ms:
                continue  # still in flight: not resting when this trade hit
            r = _classify(o, price, self.price_eps)
            if r == _NOT_REACHED:
                continue
            if r == _SWEPT_THROUGH:
                o.queue_ahead = 0.0
                f = min(o.our_qty, remaining)
                if f > self.qty_eps:
                    fills.append(FillEvent(o.price, f, o.side, ts_ms,
                                           o.placed_event_time_ms, swept=True))
                    o.our_qty -= f
                    remaining -= f
                continue
            # AT_LEVEL: deplete queue ahead first, then fill overflow into us
            d = min(o.queue_ahead, remaining)
            o.queue_ahead -= d
            remaining -= d
            if remaining > self.qty_eps and o.queue_ahead <= self.qty_eps:
                f = min(o.our_qty, remaining)
                if f > self.qty_eps:
                    fills.append(FillEvent(o.price, f, o.side, ts_ms,
                                           o.placed_event_time_ms))
                    o.our_qty -= f
                    remaining -= f

        # drop fully-filled orders so later trades cannot refill them
        self._orders = [o for o in self._orders if o.our_qty > self.qty_eps]
        return fills

    def cancel(self, oid: int) -> None:
        self._orders = [o for o in self._orders if o.id != oid]

    def cancel_all(self) -> None:
        self._orders = []

    def active_count(self) -> int:
        return len(self._orders)

    def find(self, oid: int) -> PassiveOrder | None:
        for o in self._orders:
            if o.id == oid:
                return o
        return None


def run_scenario(text: str) -> list[str]:
    """Execute a scenario DSL, return the output lines (FILL... then DONE n)."""
    qt = QueuePositionTracker()
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        op = tok[0]
        if op == "PRICE_EPS":
            qt.price_eps = float(tok[1])
        elif op == "QTY_EPS":
            qt.qty_eps = float(tok[1])
        elif op == "LATENCY":
            qt.order_latency_ms = int(tok[1])
        elif op == "ORDER":
            qt.place(int(tok[1]), float(tok[2]), float(tok[3]),
                     float(tok[4]), int(tok[5]))
        elif op == "TRADE":
            fills = qt.on_trade(float(tok[1]), float(tok[2]),
                                int(tok[3]), int(tok[4]))
            for f in fills:
                out.append(f"FILL {f.side} {f.price:.6f} {f.qty:.6f} {f.ts_ms}")
        elif op == "CANCEL":
            qt.cancel(int(tok[1]))
        # unknown ops ignored
    out.append(f"DONE {qt.active_count()}")
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def main() -> None:
    p = argparse.ArgumentParser(description="Last-in-queue fill core (Python twin).")
    p.add_argument("--run", action="store_true", help="run scenario DSL from stdin")
    args = p.parse_args()
    if args.run:
        for ln in run_scenario(sys.stdin.read()):
            print(ln)
        return
    p.print_help()


if __name__ == "__main__":
    main()
