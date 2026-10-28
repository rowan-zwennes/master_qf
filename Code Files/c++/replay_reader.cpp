// Binary replay input reader.

// HFTR v1 loader; see export_replay_binary.py for the format contract.

#include "replay_reader.hpp"

#include <cstdio>
#include <cstring>

namespace hft {

namespace {

constexpr char kMagic[4] = {'H', 'F', 'T', 'R'};
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kHeaderSize = 128;

template <typename T>
bool read_vec(std::FILE* f, std::vector<T>& out, std::size_t n) {
    out.resize(n);
    if (n == 0) return true;
    return std::fread(out.data(), sizeof(T), n, f) == n;
}

template <typename T>
T read_le(const unsigned char*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

} // namespace

bool ReplayReader::open(std::string_view path) {
    std::FILE* f = std::fopen(std::string(path).c_str(), "rb");
    if (!f) { err_ = "cannot open replay file"; return false; }
    unsigned char hdr[kHeaderSize];
    if (std::fread(hdr, 1, kHeaderSize, f) != kHeaderSize) {
        err_ = "replay header truncated"; std::fclose(f); return false;
    }
    const unsigned char* p = hdr;
    if (std::memcmp(p, kMagic, 4) != 0) {
        err_ = "bad replay magic"; std::fclose(f); return false;
    }
    p += 4;
    if (read_le<std::uint32_t>(p) != kVersion) {
        err_ = "unsupported replay version"; std::fclose(f); return false;
    }
    ReplayDay& d = day_;
    d.n_ticks = read_le<std::uint32_t>(p);
    d.n_trades = read_le<std::uint32_t>(p);
    d.n_levels = read_le<std::uint32_t>(p);
    d.n_settle = read_le<std::uint32_t>(p);
    d.n_pause = read_le<std::uint32_t>(p);
    (void)read_le<std::uint32_t>(p);    // pad
    d.t0_ms = read_le<std::int64_t>(p);
    d.tick = read_le<double>(p);
    d.quote_size = read_le<double>(p);

    const std::size_t n = d.n_ticks, m = d.n_trades;
    const std::size_t nl = n * d.n_levels;
    bool ok = read_vec(f, d.tick_ts, n) && read_vec(f, d.valid, n)
        && read_vec(f, d.bid_p, nl) && read_vec(f, d.bid_q, nl)
        && read_vec(f, d.ask_p, nl) && read_vec(f, d.ask_q, nl)
        && read_vec(f, d.mid, n) && read_vec(f, d.micro, n)
        && read_vec(f, d.alpha, n) && read_vec(f, d.alpha_ar, n)
        && read_vec(f, d.sigma, n)
        && read_vec(f, d.A, n) && read_vec(f, d.k, n)
        && read_vec(f, d.f_rate, n) && read_vec(f, d.u_s, n)
        && read_vec(f, d.trade_ts, m) && read_vec(f, d.trade_px, m)
        && read_vec(f, d.trade_qty, m) && read_vec(f, d.trade_side, m)
        && read_vec(f, d.settle_ts, d.n_settle)
        && read_vec(f, d.settle_mark, d.n_settle)
        && read_vec(f, d.settle_rate, d.n_settle)
        && read_vec(f, d.pause, 2 * static_cast<std::size_t>(d.n_pause));
    // exactly at EOF?
    const bool tail_clean = std::fgetc(f) == EOF;
    std::fclose(f);
    if (!ok) { err_ = "replay section truncated"; return false; }
    if (!tail_clean) { err_ = "trailing bytes after replay sections"; return false; }
    if (n == 0 || d.tick <= 0.0 || d.quote_size <= 0.0) {
        err_ = "degenerate replay header"; return false;
    }
    return true;
}

} // namespace hft
