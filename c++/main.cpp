// hft_engine: offline replay and latency benchmark CLI.
//
//   hft_engine replay --data day.hftr [--lut lut.bin] [--strategies 1,2,3,4]
//                     [--latency-out lat.hftb]
//   hft_engine bench  [--lut lut.bin] [--out lat.hftb]
//                     [--batches 2000] [--batch-size 512]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "glt_asymptotic.hpp"
#include "latency_bench.hpp"
#include "regime1_reader.hpp"
#include "replay_reader.hpp"
#include "riccati_lut.hpp"
#include "strategy.hpp"

namespace {

const char* arg_value(int argc, char** argv, const char* key,
                      const char* fallback = nullptr) {
    for (int i = 0; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

std::vector<int> parse_sids(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

bool load_lut_timeline(const std::string& dir, hft::LutTimeline& tl,
                       std::string& err) {
    std::ifstream f(dir + "/lut_timeline.txt");
    if (!f) { err = "cannot open " + dir + "/lut_timeline.txt"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::int64_t ts;
        std::string fname;
        if (!(ss >> ts >> fname)) { err = "bad timeline line: " + line;
                                    return false; }
        tl.ts.push_back(ts);
        tl.paths.push_back(dir + "/" + fname);
    }
    return true;
}

int run_replay(int argc, char** argv) {
    const char* data = arg_value(argc, argv, "--data");
    if (!data) { std::fprintf(stderr, "replay: --data required\n"); return 2; }
    const char* lut_path = arg_value(argc, argv, "--lut");
    const char* lut_timeline_dir = arg_value(argc, argv, "--lut-timeline");
    const char* regime1_path = arg_value(argc, argv, "--regime1");
    const char* sids_s = arg_value(argc, argv, "--strategies", "1,2,3,4");
    const char* lat_out = arg_value(argc, argv, "--latency-out");
    const char* fills_out_path = arg_value(argc, argv, "--fills-out");
    const char* ml_mode_s = arg_value(argc, argv, "--ml-mode", "horizon");

    hft::ReplayReader rr;
    if (!rr.open(data)) {
        std::fprintf(stderr, "replay: %s\n", rr.last_error().c_str());
        return 1;
    }
    hft::RiccatiLUT lut;
    const hft::RiccatiLUT* lut_p = nullptr;
    if (lut_path) {
        if (!lut.load(lut_path)) {
            std::fprintf(stderr, "lut: %s\n", lut.last_error().c_str());
            return 1;
        }
        lut_p = &lut;
    }
    hft::LutTimeline timeline;
    const hft::LutTimeline* timeline_p = nullptr;
    if (lut_timeline_dir) {
        std::string err;
        if (!load_lut_timeline(lut_timeline_dir, timeline, err)) {
            std::fprintf(stderr, "lut-timeline: %s\n", err.c_str());
            return 1;
        }
        timeline_p = &timeline;
    }
    hft::Regime1Reader r1_reader;
    const hft::Regime1Reader* r1_p = nullptr;
    if (regime1_path) {
        if (!r1_reader.load(regime1_path)) {
            std::fprintf(stderr, "regime1: %s\n",
                         r1_reader.last_error().c_str());
            return 1;
        }
        r1_p = &r1_reader;
    }
    hft::EngineConfig cfg;  // defaults mirror SimConfig; parity kit uses them
    if (std::strcmp(ml_mode_s, "defensive") == 0)
        cfg.ml_mode = hft::EngineConfig::MlMode::Defensive;
    hft::ReplayEngine eng(rr.day(), cfg, lut_p, timeline_p, r1_p);
    std::vector<std::uint64_t> lat;
    std::vector<hft::FillRecord> fills;
    const auto res =
        eng.run(parse_sids(sids_s), lat_out ? &lat : nullptr,
                fills_out_path ? &fills : nullptr);

    if (fills_out_path) {
        std::FILE* ff = std::fopen(fills_out_path, "w");
        if (!ff) { std::fprintf(stderr, "cannot open %s\n", fills_out_path);
                   return 1; }
        std::fprintf(ff, "sid,ts_ms,side,price,qty,fee,inv_after,swept\n");
        for (const auto& f : fills)
            std::fprintf(ff, "%d,%lld,%d,%.10f,%.10f,%.10f,%.10f,%d\n",
                         f.sid, static_cast<long long>(f.ts_ms), f.side,
                         f.price, f.qty, f.fee, f.inv_after, f.swept ? 1 : 0);
        std::fclose(ff);
    }

    std::printf("{\n");
    for (std::size_t i = 0; i < res.size(); ++i) {
        const auto& r = res[i];
        std::printf("  \"%d\": {\"terminal_pnl\": %.10f, \"fees\": %.10f, "
                    "\"funding\": %.10f, \"n_fills\": %ld, "
                    "\"mean_abs_inv\": %.10f, \"frac_time_at_cap\": %.10f, "
                    "\"downtime_s\": %.3f}%s\n",
                    r.sid, r.terminal_pnl, r.fees, r.funding, r.n_fills,
                    r.mean_abs_inv, r.frac_time_at_cap, r.downtime_s,
                    i + 1 < res.size() ? "," : "");
    }
    std::printf("}\n");

    if (lat_out) {
        std::vector<hft::StageSamples> stages(1);
        stages[0].name = "replay_quote_e2e";
        stages[0].ns = std::move(lat);
        if (!hft::latency_dump(lat_out, stages)) {
            std::fprintf(stderr, "latency dump failed\n");
            return 1;
        }
    }
    return 0;
}

int run_lutprobe(int argc, char** argv) {
    const char* lut_path = arg_value(argc, argv, "--lut");
    const char* qs = arg_value(argc, argv, "--q");
    const char* us = arg_value(argc, argv, "--u");
    if (!lut_path || !qs || !us) {
        std::fprintf(stderr, "lutprobe: --lut --q q1,q2 --u u1,u2 required\n");
        return 2;
    }
    hft::RiccatiLUT lut;
    if (!lut.load(lut_path)) {
        std::fprintf(stderr, "lut: %s\n", lut.last_error().c_str());
        return 1;
    }
    const char* q0s = arg_value(argc, argv, "--q0");
    const bool with_drift = q0s != nullptr;
    const double q0 = with_drift ? std::atof(q0s) : 0.0;
    const auto qv = parse_sids(qs);
    std::vector<double> uv;
    for (const char* p = us; *p;) {
        uv.push_back(std::atof(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    auto num = [](double v, char* buf, std::size_t n) -> const char* {
        if (std::isnan(v)) { std::snprintf(buf, n, "NaN"); return buf; }
        std::snprintf(buf, n, "%.10f", v);
        return buf;
    };
    std::printf("{\"u_star_s\": %.10f, \"has_sens\": %s, \"q0\": %.10f, "
                "\"probes\": [\n",
                lut.u_star_seconds(), lut.has_sens() ? "true" : "false", q0);
    bool first = true;
    char bb[32], ab[32];
    for (int q : qv)
        for (double u : uv) {
            const auto d = lut.depths(q, u);
            std::printf("%s  {\"q\": %d, \"u\": %.6f, \"db\": %s, "
                        "\"da\": %s",
                        first ? "" : ",\n", q, u,
                        num(d.bid, bb, sizeof bb), num(d.ask, ab, sizeof ab));
            if (with_drift) {
                const auto dd = lut.depths_drift(q, u, q0);
                std::printf(", \"db_drift\": %s, \"da_drift\": %s",
                            num(dd.bid, bb, sizeof bb),
                            num(dd.ask, ab, sizeof ab));
            }
            std::printf("}");
            first = false;
        }
    std::printf("\n]}\n");
    return 0;
}

int run_r1probe(int argc, char** argv) {
    const char* man_path = arg_value(argc, argv, "--regime1");
    const char* qs = arg_value(argc, argv, "--q");
    const char* ts = arg_value(argc, argv, "--ts");
    if (!man_path || !qs || !ts) {
        std::fprintf(stderr,
                     "r1probe: --regime1 man.txt --q q1,q2 --ts t1,t2 "
                     "[--stream auto|ar|ml] [--gamma --sigma --A --k]\n");
        return 2;
    }
    hft::Regime1Reader r1;
    if (!r1.load(man_path)) {
        std::fprintf(stderr, "regime1: %s\n", r1.last_error().c_str());
        return 1;
    }
    const char* sname = arg_value(argc, argv, "--stream", "auto");
    using S = hft::Regime1Reader::Stream;
    const S stream = (std::strcmp(sname, "ar") == 0) ? S::Ar
                     : (std::strcmp(sname, "ml") == 0) ? S::Ml
                                                       : S::Auto;
    if (!r1.ready(stream)) {
        std::fprintf(stderr, "regime1: stream '%s' empty\n", sname);
        return 1;
    }
    const hft::glt::MarketConsts mc(
        std::atof(arg_value(argc, argv, "--gamma", "2.0e-5")),
        std::atof(arg_value(argc, argv, "--sigma", "4.575")),
        std::atof(arg_value(argc, argv, "--A", "0.2742")),
        std::atof(arg_value(argc, argv, "--k", "0.0900")));
    const auto qv = parse_sids(qs);
    std::vector<std::int64_t> tv;
    for (const char* p = ts; *p;) {
        tv.push_back(static_cast<std::int64_t>(std::atoll(p)));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    auto num = [](double v, char* buf, std::size_t n) -> const char* {
        if (std::isnan(v)) { std::snprintf(buf, n, "NaN"); return buf; }
        std::snprintf(buf, n, "%.12f", v);
        return buf;
    };
    std::printf("{\"stream\": \"%s\", \"Q\": %d, \"ts_first\": %lld, "
                "\"ts_last\": %lld, \"probes\": [\n",
                sname, r1.Q(), static_cast<long long>(r1.ts_first(stream)),
                static_cast<long long>(r1.ts_last(stream)));
    bool first = true;
    char bb[32], ab[32];
    std::sort(tv.begin(), tv.end());
    for (int q : qv)
        for (std::int64_t t : tv) {
            const auto d = r1.depths(stream, t, q, mc);
            std::printf("%s  {\"q\": %d, \"ts\": %lld, \"db\": %s, \"da\": %s}",
                        first ? "" : ",\n", q, static_cast<long long>(t),
                        num(d.bid, bb, sizeof bb), num(d.ask, ab, sizeof ab));
            first = false;
        }
    std::printf("\n]}\n");
    return 0;
}

int run_bench(int argc, char** argv) {
    const char* lut_path = arg_value(argc, argv, "--lut");
    const char* regime1_path = arg_value(argc, argv, "--regime1");
    const char* out = arg_value(argc, argv, "--out", "latency.hftb");
    const std::size_t batches = static_cast<std::size_t>(
        std::atol(arg_value(argc, argv, "--batches", "2000")));
    const std::size_t batch_size = static_cast<std::size_t>(
        std::atol(arg_value(argc, argv, "--batch-size", "512")));

    std::vector<hft::StageSamples> stages;
    {
        hft::StageSamples s;
        s.name = "glt_closed_form";
        s.ns_per_op = hft::bench_glt_closed_form(batches, batch_size);
        stages.push_back(std::move(s));
    }
    hft::RiccatiLUT lut;
    if (lut_path) {
        if (!lut.load(lut_path)) {
            std::fprintf(stderr, "lut: %s\n", lut.last_error().c_str());
            return 1;
        }
        hft::StageSamples s;
        s.name = "lut_interp";
        s.ns_per_op = hft::bench_lut_interp(lut, batches, batch_size);
        stages.push_back(std::move(s));
    }
    hft::Regime1Reader r1_reader;
    if (regime1_path) {
        if (!r1_reader.load(regime1_path)) {
            std::fprintf(stderr, "regime1: %s\n",
                         r1_reader.last_error().c_str());
            return 1;
        }
        if (!r1_reader.ready(hft::Regime1Reader::Stream::Auto)) {
            std::fprintf(stderr, "regime1: no autonomous stream in manifest\n");
            return 1;
        }
        hft::StageSamples s;
        s.name = "regime1_read";
        s.ns_per_op = hft::bench_regime1_read(r1_reader, batches, batch_size);
        stages.push_back(std::move(s));
    }
    if (!hft::latency_dump(out, stages)) {
        std::fprintf(stderr, "latency dump failed\n");
        return 1;
    }
    std::printf("{\"stages\": %zu, \"batches\": %zu, \"batch_size\": %zu, "
                "\"out\": \"%s\"}\n",
                stages.size(), batches, batch_size, out);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: hft_engine replay --data day.hftr "
                     "[--lut ...] [--lut-timeline dir] [--regime1 manifest] "
                     "[...]\n"
                     "       hft_engine bench [--lut lut.bin] "
                     "[--regime1 manifest] [...]\n");
        return 2;
    }
    if (std::strcmp(argv[1], "replay") == 0) return run_replay(argc, argv);
    if (std::strcmp(argv[1], "bench") == 0) return run_bench(argc, argv);
    if (std::strcmp(argv[1], "lutprobe") == 0) return run_lutprobe(argc, argv);
    if (std::strcmp(argv[1], "r1probe") == 0) return run_r1probe(argc, argv);
    std::fprintf(stderr, "unknown mode '%s'\n", argv[1]);
    return 2;
}
