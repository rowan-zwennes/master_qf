// Scenario runner for the fill core, mirrored by fill_core_reference.py.

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "queue_position.hpp"

int main() {
    hft::QueuePositionTracker qt;
    std::string line;
    while (std::getline(std::cin, line)) {
        // strip comments
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ss(line);
        std::string op;
        if (!(ss >> op)) continue;  // blank line

        if (op == "PRICE_EPS") {
            double v;
            ss >> v;
            qt.set_price_eps(v);
        } else if (op == "QTY_EPS") {
            double v;
            ss >> v;
            qt.set_qty_eps(v);
        } else if (op == "ORDER") {
            int side;
            double price, qty, qa;
            std::uint64_t ts;
            ss >> side >> price >> qty >> qa >> ts;
            qt.place(side, price, qty, qa, ts);
        } else if (op == "TRADE") {
            double price, qty;
            int taker;
            std::uint64_t ts;
            ss >> price >> qty >> taker >> ts;
            const auto fills = qt.on_trade(price, qty, taker, ts);
            for (const auto& f : fills) {
                std::printf("FILL %d %.6f %.6f %llu\n", f.side, f.price, f.qty,
                            static_cast<unsigned long long>(f.ts_ms));
            }
        } else if (op == "CANCEL") {
            int id;
            ss >> id;
            qt.cancel(id);
        } else if (op == "LATENCY") {
            std::uint64_t l;
            ss >> l;
            qt.set_order_latency_ms(l);
        }
        // unknown ops ignored
    }
    std::printf("DONE %zu\n", qt.active_count());
    return 0;
}
