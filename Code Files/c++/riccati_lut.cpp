// Regime II lookup-table reader and interpolator.

#include "riccati_lut.hpp"

#include <sys/mman.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace hft {

namespace {

constexpr std::uint32_t kHeaderSize = 256;
constexpr std::uint32_t kRowAlign = 64;
constexpr char kMagic[4] = {'H', 'F', 'T', 'L'};
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kSensHeaderSize = 64;
constexpr char kSensMagic[4] = {'H', 'F', 'T', 'S'};
constexpr std::uint32_t kSensVersion = 1;

// CRC32 (IEEE, zlib-compatible).
std::uint32_t crc32_ieee(const unsigned char* p, std::size_t n) {
    static std::uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i)
        c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

template <typename T>
T read_le(const unsigned char*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

double* page_alloc(std::size_t bytes) {
    return static_cast<double*>(
        std::aligned_alloc(4096, ((bytes + 4095) / 4096) * 4096));
}

} // namespace

struct RiccatiLUT::Mapping {
    LutHeader hdr{};
    double du_s{};
    int Q{};
    std::uint32_t n_u{};
    double* delta_b{nullptr};
    double* delta_a{nullptr};
    float* logw{nullptr};               // raw log-omega (parity checks)
    std::size_t bytes_b{}, bytes_l{};
    double* db_sens{nullptr};
    double* da_sens{nullptr};
    bool has_sens{false};
};

RiccatiLUT::~RiccatiLUT() {
    destroy_mapping(const_cast<Mapping*>(active_.load()));
}

void RiccatiLUT::destroy_mapping(Mapping* m) {
    if (!m) return;
    if (m->delta_b) { munlock(m->delta_b, m->bytes_b); std::free(m->delta_b); }
    if (m->delta_a) { munlock(m->delta_a, m->bytes_b); std::free(m->delta_a); }
    if (m->logw) { munlock(m->logw, m->bytes_l); std::free(m->logw); }
    if (m->db_sens) { munlock(m->db_sens, m->bytes_b); std::free(m->db_sens); }
    if (m->da_sens) { munlock(m->da_sens, m->bytes_b); std::free(m->da_sens); }
    delete m;
}

namespace {
bool load_sens_companion(const std::string& lut_path, std::uint32_t n_q,
                         std::uint32_t n_u, double* db_sens, double* da_sens) {
    std::FILE* f = std::fopen((lut_path + ".sens").c_str(), "rb");
    if (!f) return false;
    unsigned char hb[kSensHeaderSize];
    if (std::fread(hb, 1, kSensHeaderSize, f) != kSensHeaderSize) {
        std::fclose(f); return false;
    }
    const unsigned char* p = hb;
    if (std::memcmp(p, kSensMagic, 4) != 0) { std::fclose(f); return false; }
    p += 4;
    if (read_le<std::uint32_t>(p) != kSensVersion) { std::fclose(f); return false; }
    const auto s_nq = read_le<std::uint32_t>(p);
    const auto s_nu = read_le<std::uint32_t>(p);
    (void)read_le<std::uint32_t>(p);                       // du_ms
    (void)read_le<std::int32_t>(p);                        // q_min
    (void)read_le<std::int32_t>(p);                        // q_max
    (void)read_le<double>(p);                              // q0_ref
    const auto crc = read_le<std::uint32_t>(p);
    if (s_nq != n_q || s_nu != n_u) { std::fclose(f); return false; }
    const std::size_t row_bytes = static_cast<std::size_t>(n_u) * 4;
    const std::size_t padded =
        ((row_bytes + kRowAlign - 1) / kRowAlign) * kRowAlign;
    const std::size_t body_bytes = padded * 2 * n_q;       // db_sens then da_sens
    std::vector<unsigned char> body(body_bytes);
    const std::size_t got = std::fread(body.data(), 1, body_bytes, f);
    std::fclose(f);
    if (got != body_bytes) return false;
    if (crc != 0 && crc32_ieee(body.data(), body_bytes) != crc) return false;
    for (std::uint32_t r = 0; r < n_q; ++r) {
        const float* sb = reinterpret_cast<const float*>(
            body.data() + static_cast<std::size_t>(r) * padded);
        const float* sa = reinterpret_cast<const float*>(
            body.data() + static_cast<std::size_t>(n_q + r) * padded);
        double* ob = db_sens + static_cast<std::size_t>(r) * n_u;
        double* oa = da_sens + static_cast<std::size_t>(r) * n_u;
        for (std::uint32_t j = 0; j < n_u; ++j) {
            ob[j] = static_cast<double>(sb[j]);
            oa[j] = static_cast<double>(sa[j]);
        }
    }
    return true;
}
} // namespace

RiccatiLUT::Mapping* RiccatiLUT::build_mapping(std::string_view path,
                                               std::string& err) {
    std::FILE* f = std::fopen(std::string(path).c_str(), "rb");
    if (!f) { err = "cannot open LUT file"; return nullptr; }
    unsigned char hdr_buf[kHeaderSize];
    if (std::fread(hdr_buf, 1, kHeaderSize, f) != kHeaderSize) {
        err = "LUT header truncated"; std::fclose(f); return nullptr;
    }
    const unsigned char* p = hdr_buf;
    if (std::memcmp(p, kMagic, 4) != 0) {
        err = "bad LUT magic"; std::fclose(f); return nullptr;
    }
    p += 4;
    const auto version = read_le<std::uint32_t>(p);
    if (version != kVersion) {
        err = "unsupported LUT version"; std::fclose(f); return nullptr;
    }
    LutHeader h{};
    h.n_q = read_le<std::uint32_t>(p);
    h.n_u = read_le<std::uint32_t>(p);
    h.du_ms = read_le<std::uint32_t>(p);
    h.q_min = read_le<std::int32_t>(p);
    h.q_max = read_le<std::int32_t>(p);
    (void)read_le<std::uint32_t>(p);    // header pad
    h.gamma = read_le<double>(p);
    h.k = read_le<double>(p);
    h.A = read_le<double>(p);
    h.rho = read_le<double>(p);
    h.f_t = read_le<double>(p);
    h.F_t = read_le<double>(p);
    h.sigma = read_le<double>(p);
    h.alpha_ml = read_le<double>(p);
    h.u_star_s = read_le<double>(p);
    h.epoch_index = read_le<std::uint32_t>(p);
    h.body_crc32 = read_le<std::uint32_t>(p);
    h.build_ts_ns = read_le<std::uint64_t>(p);

    const int Q = h.q_max;
    if (h.q_min != -Q || h.n_q != static_cast<std::uint32_t>(2 * Q + 1)
        || h.n_u < 2 || h.du_ms == 0) {
        err = "inconsistent LUT geometry"; std::fclose(f); return nullptr;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(h.n_u) * 4;
    const std::size_t padded =
        ((row_bytes + kRowAlign - 1) / kRowAlign) * kRowAlign;
    const std::size_t body_bytes = padded * h.n_q;

    std::vector<unsigned char> body(body_bytes);
    const std::size_t got = std::fread(body.data(), 1, body_bytes, f);
    std::fclose(f);
    if (got != body_bytes) { err = "LUT body truncated"; return nullptr; }
    if (h.body_crc32 != 0
        && crc32_ieee(body.data(), body_bytes) != h.body_crc32) {
        err = "LUT body CRC mismatch"; return nullptr;
    }

    auto* m = new Mapping{};
    m->hdr = h;
    m->du_s = h.du_ms / 1000.0;
    m->Q = Q;
    m->n_u = h.n_u;
    const std::size_t grid = static_cast<std::size_t>(h.n_q) * h.n_u;
    m->bytes_b = grid * sizeof(double);
    m->bytes_l = grid * sizeof(float);
    m->delta_b = page_alloc(m->bytes_b);
    m->delta_a = page_alloc(m->bytes_b);
    m->logw = reinterpret_cast<float*>(page_alloc(m->bytes_l));
    if (!m->delta_b || !m->delta_a || !m->logw) {
        err = "LUT allocation failed"; destroy_mapping(m); return nullptr;
    }

    for (std::uint32_t r = 0; r < h.n_q; ++r)
        std::memcpy(m->logw + static_cast<std::size_t>(r) * h.n_u,
                    body.data() + r * padded, row_bytes);
    const double c1 = (1.0 / h.gamma) * std::log1p(h.gamma / h.k);
    const double inv_k = 1.0 / h.k;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::uint32_t r = 0; r < h.n_q; ++r) {
        const int q = Q - static_cast<int>(r);
        const float* y_q = m->logw + static_cast<std::size_t>(r) * h.n_u;
        const float* y_up = (q != Q)        // row of q+1 sits ABOVE (r-1)
            ? m->logw + static_cast<std::size_t>(r - 1) * h.n_u : nullptr;
        const float* y_dn = (q != -Q)       // row of q-1 sits BELOW (r+1)
            ? m->logw + static_cast<std::size_t>(r + 1) * h.n_u : nullptr;
        double* db = m->delta_b + static_cast<std::size_t>(r) * h.n_u;
        double* da = m->delta_a + static_cast<std::size_t>(r) * h.n_u;
        for (std::uint32_t j = 0; j < h.n_u; ++j) {
            db[j] = y_up ? c1 + inv_k * (static_cast<double>(y_q[j])
                                         - static_cast<double>(y_up[j]))
                         : nan;
            da[j] = y_dn ? c1 + inv_k * (static_cast<double>(y_q[j])
                                         - static_cast<double>(y_dn[j]))
                         : nan;
        }
    }
    // Optional linear-response drift sensitivity companion (<path>.sens).
    double* db_sens = page_alloc(m->bytes_b);
    double* da_sens = page_alloc(m->bytes_b);
    if (db_sens && da_sens
        && load_sens_companion(std::string(path), h.n_q, h.n_u,
                               db_sens, da_sens)) {
        m->db_sens = db_sens;
        m->da_sens = da_sens;
        m->has_sens = true;
        (void)mlock(m->db_sens, m->bytes_b);
        (void)mlock(m->da_sens, m->bytes_b);
    } else {
        std::free(db_sens);
        std::free(da_sens);
    }

    (void)mlock(m->delta_b, m->bytes_b);
    (void)mlock(m->delta_a, m->bytes_b);
    (void)mlock(m->logw, m->bytes_l);
    return m;
}

bool RiccatiLUT::load(std::string_view path) {
    Mapping* m = build_mapping(path, err_);
    if (!m) return false;
    const Mapping* old = active_.exchange(m, std::memory_order_acq_rel);
    destroy_mapping(const_cast<Mapping*>(old));
    return true;
}

bool RiccatiLUT::reload_and_swap(std::string_view path) { return load(path); }

const LutHeader& RiccatiLUT::header() const noexcept {
    return active_.load(std::memory_order_acquire)->hdr;
}

double RiccatiLUT::u_star_seconds() const noexcept {
    const Mapping* m = active_.load(std::memory_order_acquire);
    return m ? m->hdr.u_star_s : -1.0;
}

RiccatiLUT::D2 RiccatiLUT::depths(int q, double u_seconds) const noexcept {
    const Mapping* m = active_.load(std::memory_order_acquire);
    const std::size_t r =
        static_cast<std::size_t>(m->Q - q) * m->n_u;     // row offset
    const double x = u_seconds / m->du_s;
    if (x <= 0.0) return {m->delta_b[r], m->delta_a[r]};
    if (x >= static_cast<double>(m->n_u - 1))
        return {m->delta_b[r + m->n_u - 1], m->delta_a[r + m->n_u - 1]};
    const auto i0 = static_cast<std::uint32_t>(x);
    const double a = x - i0;
    return {(1.0 - a) * m->delta_b[r + i0] + a * m->delta_b[r + i0 + 1],
            (1.0 - a) * m->delta_a[r + i0] + a * m->delta_a[r + i0 + 1]};
}

bool RiccatiLUT::has_sens() const noexcept {
    const Mapping* m = active_.load(std::memory_order_acquire);
    return m && m->has_sens;
}

RiccatiLUT::D2 RiccatiLUT::depths_drift(int q, double u_seconds,
                                        double q0) const noexcept {
    const Mapping* m = active_.load(std::memory_order_acquire);
    const std::size_t r = static_cast<std::size_t>(m->Q - q) * m->n_u;
    const double x = u_seconds / m->du_s;
    // base LUT depths (same interpolation as depths()).
    double db, da;
    std::size_t i0;
    double a;
    if (x <= 0.0) { i0 = 0; a = 0.0; }
    else if (x >= static_cast<double>(m->n_u - 1)) { i0 = m->n_u - 1; a = 0.0; }
    else { i0 = static_cast<std::uint32_t>(x); a = x - i0; }
    const std::size_t j0 = r + i0;
    const std::size_t j1 = (a == 0.0) ? j0 : j0 + 1;
    db = (1.0 - a) * m->delta_b[j0] + a * m->delta_b[j1];
    da = (1.0 - a) * m->delta_a[j0] + a * m->delta_a[j1];
    if (!m->has_sens) return {db, da};
    const double sb = (1.0 - a) * m->db_sens[j0] + a * m->db_sens[j1];
    const double sa = (1.0 - a) * m->da_sens[j0] + a * m->da_sens[j1];
    return {db + q0 * sb, da + q0 * sa};
}

double RiccatiLUT::log_omega(int q, double u_seconds) const noexcept {
    const Mapping* m = active_.load(std::memory_order_acquire);
    const std::size_t r = static_cast<std::size_t>(m->Q - q) * m->n_u;
    const double x = u_seconds / m->du_s;
    if (x <= 0.0) return m->logw[r];
    if (x >= static_cast<double>(m->n_u - 1)) return m->logw[r + m->n_u - 1];
    const auto i0 = static_cast<std::uint32_t>(x);
    const double a = x - i0;
    return (1.0 - a) * m->logw[r + i0] + a * m->logw[r + i0 + 1];
}

} // namespace hft
