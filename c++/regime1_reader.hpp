// Reader for the Regime I eigenvector table.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "glt_asymptotic.hpp"   // glt::MarketConsts (c1, k)

namespace hft {

class Regime1Reader {
public:
    enum class Stream { Auto = 0, Ar = 1, Ml = 2 };
    struct D2 { double bid, ask; };

    static constexpr std::uint32_t kMagic = 0x52314631u;  // 'R1F1'

    // Map a strategy's drift descriptor to its Regime-I stream.
    static Stream stream_for(bool use_ml, bool use_ar) noexcept {
        if (use_ml) return Stream::Ml;
        if (use_ar) return Stream::Ar;
        return Stream::Auto;
    }

    bool load(std::string_view manifest_path) {
        std::ifstream man{std::string(manifest_path)};
        if (!man) { err_ = "cannot open manifest"; return false; }
        const std::string dir = parent_dir(manifest_path);
        std::string tok;
        man >> tok;                                   // "n_q"
        if (tok != "n_q") { err_ = "expected n_q header"; return false; }
        man >> n_q_;
        if (n_q_ == 0 || (n_q_ % 2) == 0) { err_ = "n_q must be odd, >0"; return false; }
        Q_ = (static_cast<int>(n_q_) - 1) / 2;
        std::string name, file;
        std::uint32_t n_rec = 0;
        while (man >> name >> file >> n_rec) {
            if (name == "hdrift") {
                if (!read_hdrift_stream(dir + file)) return false;
                continue;
            }
            Stream s;
            if (name == "auto") s = Stream::Auto;
            else if (name == "ar") s = Stream::Ar;
            else if (name == "ml") s = Stream::Ml;
            else { err_ = "unknown stream '" + name + "'"; return false; }
            if (!read_stream(dir + file, streams_[idx(s)])) return false;
        }
        return true;
    }

    double hdrift(std::int64_t ts_ms) const noexcept {
        const std::size_t n = hd_ts_.size();
        if (n == 0) return std::numeric_limits<double>::quiet_NaN();
        std::size_t c = hd_cursor_;
        if (c >= n) c = 0;
        if (hd_ts_[c] <= ts_ms) {
            while (c + 1 < n && hd_ts_[c + 1] <= ts_ms) ++c;
        } else {
            std::size_t lo = 0, hi = c;
            while (lo < hi) {
                const std::size_t mid = lo + ((hi - lo) >> 1);
                if (hd_ts_[mid] <= ts_ms) lo = mid + 1; else hi = mid;
            }
            c = (lo == 0) ? 0 : lo - 1;
        }
        hd_cursor_ = c;
        return hd_v_[c];
    }

    bool ready(Stream s) const noexcept { return !streams_[idx(s)].ts.empty(); }
    int Q() const noexcept { return Q_; }
    const std::string& last_error() const noexcept { return err_; }

    std::int64_t ts_first(Stream s) const noexcept {
        const auto& v = streams_[idx(s)].ts; return v.empty() ? 0 : v.front();
    }
    std::int64_t ts_last(Stream s) const noexcept {
        const auto& v = streams_[idx(s)].ts; return v.empty() ? 0 : v.back();
    }

    D2 depths(Stream s, std::int64_t ts_ms, int q,
              const glt::MarketConsts& mc) const noexcept {
        const std::size_t si = idx(s);
        const Stream2& st = streams_[si];
        const std::size_t n = st.ts.size();
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        if (n == 0) return {nan, nan};
        const std::size_t row = asof_row(st, si, ts_ms);
        const float* lf = &st.log_f0[row * n_q_];
        const int p = Q_ - q;                          // q = Q..-Q index
        const double inv_k = 1.0 / mc.k;
        // delta_b uses q+1 (index p-1); delta_a uses q-1 (index p+1).
        const double bid = (p >= 1)
            ? mc.c1 + inv_k * (static_cast<double>(lf[p]) - static_cast<double>(lf[p - 1]))
            : nan;
        const double ask = (p + 1 < static_cast<int>(n_q_))
            ? mc.c1 + inv_k * (static_cast<double>(lf[p]) - static_cast<double>(lf[p + 1]))
            : nan;
        return {bid, ask};
    }

private:
    struct Stream2 {
        std::vector<std::int64_t> ts;   // ascending, n_rec
        std::vector<float> log_f0;      // n_rec * n_q, q = Q..-Q order
    };

    static std::size_t idx(Stream s) noexcept { return static_cast<std::size_t>(s); }

    std::size_t asof_row(const Stream2& st, std::size_t si,
                         std::int64_t ts_ms) const noexcept {
        const std::size_t n = st.ts.size();
        std::size_t c = cursor_[si];
        if (c >= n) c = 0;
        if (st.ts[c] <= ts_ms) {
            while (c + 1 < n && st.ts[c + 1] <= ts_ms) ++c;   // advance forward
        } else {
            // ts moved backward: binary search in [0, c) for first ts > ts_ms
            std::size_t lo = 0, hi = c;
            while (lo < hi) {
                const std::size_t mid = lo + ((hi - lo) >> 1);
                if (st.ts[mid] <= ts_ms) lo = mid + 1; else hi = mid;
            }
            c = (lo == 0) ? 0 : lo - 1;                        // clamp before open
        }
        cursor_[si] = c;
        return c;
    }

    static std::string parent_dir(std::string_view p) {
        const auto pos = p.find_last_of("/\\");
        return pos == std::string_view::npos ? std::string{}
                                             : std::string(p.substr(0, pos + 1));
    }

    bool read_hdrift_stream(const std::string& path) {
        std::ifstream f{path, std::ios::binary};
        if (!f) { err_ = "cannot open stream " + path; return false; }
        std::uint32_t hdr[4];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if (!f || hdr[0] != kMagic) { err_ = "bad magic in " + path; return false; }
        if (hdr[1] != 0 || hdr[3] != 1) {
            err_ = "not an hdrift (kind-1 scalar) stream: " + path;
            return false;
        }
        const std::uint32_t n_rec = hdr[2];
        hd_ts_.resize(n_rec);
        hd_v_.resize(n_rec);
        for (std::uint32_t r = 0; r < n_rec; ++r) {
            std::int64_t ts;
            double v;
            f.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            f.read(reinterpret_cast<char*>(&v), sizeof(v));
            if (!f) { err_ = "truncated stream " + path; return false; }
            hd_ts_[r] = ts;
            hd_v_[r] = v;
        }
        return true;
    }

    bool read_stream(const std::string& path, Stream2& out) {
        std::ifstream f{path, std::ios::binary};
        if (!f) { err_ = "cannot open stream " + path; return false; }
        std::uint32_t hdr[4];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if (!f || hdr[0] != kMagic) { err_ = "bad magic in " + path; return false; }
        if (hdr[1] != n_q_) { err_ = "n_q mismatch in " + path; return false; }
        const std::uint32_t n_rec = hdr[2];
        out.ts.resize(n_rec);
        out.log_f0.resize(static_cast<std::size_t>(n_rec) * n_q_);
        for (std::uint32_t r = 0; r < n_rec; ++r) {
            std::int64_t ts;
            f.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            f.read(reinterpret_cast<char*>(&out.log_f0[static_cast<std::size_t>(r) * n_q_]),
                   static_cast<std::streamsize>(sizeof(float) * n_q_));
            if (!f) { err_ = "truncated stream " + path; return false; }
            out.ts[r] = ts;
        }
        return true;
    }

    std::uint32_t n_q_{};
    int Q_{};
    Stream2 streams_[3];
    mutable std::size_t cursor_[3]{};   // per-stream forward as-of cursor (hot path)
    std::vector<std::int64_t> hd_ts_;   // hdrift scalar timeline (may be empty)
    std::vector<double> hd_v_;
    mutable std::size_t hd_cursor_{};
    std::string err_;
};

} // namespace hft
