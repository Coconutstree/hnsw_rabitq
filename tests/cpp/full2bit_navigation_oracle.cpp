#include "full2bit_experiment_common.h"

#include <iomanip>
#include <unordered_map>

namespace {

class BaseVectorReader {
 public:
    BaseVectorReader(std::string path, size_t dim)
        : path_(std::move(path)), dim_(dim), input_(path_, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error("failed to open base fvecs: " + path_);
        }
    }

    std::vector<float> read(size_t row) {
        auto found = cache_.find(row);
        if (found != cache_.end()) {
            return found->second;
        }
        const std::streamoff record_bytes =
            static_cast<std::streamoff>(sizeof(int32_t) + dim_ * sizeof(float));
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(row) * record_bytes, std::ios::beg);
        int32_t stored_dim = 0;
        input_.read(reinterpret_cast<char *>(&stored_dim), sizeof(stored_dim));
        if (!input_ || stored_dim != static_cast<int32_t>(dim_)) {
            throw std::runtime_error("bad base fvec row");
        }
        std::vector<float> vector(dim_);
        input_.read(reinterpret_cast<char *>(vector.data()), dim_ * sizeof(float));
        if (!input_) {
            throw std::runtime_error("truncated base fvec row");
        }
        auto inserted = cache_.emplace(row, std::move(vector));
        return inserted.first->second;
    }

 private:
    std::string path_;
    size_t dim_;
    std::ifstream input_;
    std::unordered_map<size_t, std::vector<float>> cache_;
};

struct Accum {
    double recall1 = 0.0;
    double recall10 = 0.0;
    double recall100 = 0.0;
    double coverage10 = 0.0;
    double coverage100 = 0.0;
    double us = 0.0;
};

std::vector<full2bit_exp::labeltype> float_rerank(
    const float *query,
    const std::vector<std::pair<float, full2bit_exp::labeltype>> &candidates,
    BaseVectorReader &base_reader,
    size_t dim,
    size_t limit) {
    std::vector<std::pair<float, full2bit_exp::labeltype>> scored;
    scored.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        const auto base = base_reader.read(static_cast<size_t>(candidate.second));
        scored.emplace_back(full2bit_exp::l2_sqr(query, base.data(), dim), candidate.second);
    }
    std::sort(scored.begin(), scored.end());
    return full2bit_exp::labels_from_pairs(scored, limit);
}

void add_metrics(
    Accum &acc,
    const std::vector<full2bit_exp::labeltype> &result,
    const std::vector<std::pair<float, full2bit_exp::labeltype>> &candidate_pairs,
    const std::vector<full2bit_exp::labeltype> &gt,
    double elapsed_us) {
    acc.recall1 += full2bit_exp::recall_at(result, gt, 1);
    acc.recall10 += full2bit_exp::recall_at(result, gt, 10);
    acc.recall100 += full2bit_exp::recall_at(result, gt, 100);
    const auto candidate_labels =
        full2bit_exp::labels_from_pairs(candidate_pairs, candidate_pairs.size());
    auto coverage = [](const std::vector<full2bit_exp::labeltype> &pool,
                       const std::vector<full2bit_exp::labeltype> &truth,
                       size_t k) {
        std::unordered_set<full2bit_exp::labeltype> candidate_set;
        candidate_set.reserve(pool.size() * 2U);
        for (full2bit_exp::labeltype label : pool) {
            candidate_set.insert(label);
        }
        const size_t limit = std::min(k, truth.size());
        size_t hits = 0;
        for (size_t i = 0; i < limit; ++i) {
            hits += candidate_set.count(truth[i]) != 0 ? 1U : 0U;
        }
        return limit == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(limit);
    };
    acc.coverage10 += coverage(candidate_labels, gt, 10);
    acc.coverage100 += coverage(candidate_labels, gt, 100);
    acc.us += elapsed_us;
}

}  // namespace

int main() {
    using namespace full2bit_exp;
    try {
        const Config cfg = load_config();
        print_config(cfg, "navigation_oracle");
        auto index = load_full2bit_index(cfg);
        const size_t eval_queries =
            std::min(getenv_size_t("RABITQ_ORACLE_QUERIES", 100), cfg.query_count);
        const auto efs = parse_efs(
            getenv_string("RABITQ_EFS", ""),
            std::vector<size_t>{50, 100, 150, 200, 300, 460});
        const std::vector<float> queries = read_fvecs(cfg.query_path, eval_queries, cfg.dim);
        const auto gt = read_ivecs(cfg.gt_path, eval_queries, cfg.gt_width);
        BaseVectorReader base_reader(cfg.base_path, cfg.dim);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mode\tef\tcandidate_count\tRecall@1\tRecall@10\tRecall@100\t"
                     "CandidateCoverage@10\tCandidateCoverage@100\tAvgUs\n";
        for (size_t ef : efs) {
            index->setEf(ef);
            Accum full2bit_return;
            Accum float_top100;
            Accum float_top500;
            for (size_t qi = 0; qi < eval_queries; ++qi) {
                const float *query = queries.data() + qi * cfg.dim;

                auto begin = std::chrono::steady_clock::now();
                auto q100 = index->index().searchKnn(query, 100);
                const double us100 = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count();
                const auto candidates100 = queue_to_sorted(q100);
                const auto full2bit_labels = labels_from_pairs(candidates100, 100);
                add_metrics(full2bit_return, full2bit_labels, candidates100, gt[qi], us100);

                begin = std::chrono::steady_clock::now();
                const auto float100 = float_rerank(query, candidates100, base_reader, cfg.dim, 100);
                const double rerank100_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count() + us100;
                add_metrics(float_top100, float100, candidates100, gt[qi], rerank100_us);

                begin = std::chrono::steady_clock::now();
                auto q500 = index->index().searchKnn(query, 500);
                const double search500_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count();
                const auto candidates500 = queue_to_sorted(q500);
                begin = std::chrono::steady_clock::now();
                const auto float500 = float_rerank(query, candidates500, base_reader, cfg.dim, 100);
                const double total500_us = search500_us + std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count();
                add_metrics(float_top500, float500, candidates500, gt[qi], total500_us);
            }

            auto print_row = [&](const char *mode, size_t candidate_count, const Accum &acc) {
                const double n = static_cast<double>(eval_queries);
                std::cout << mode << "\t" << ef << "\t" << candidate_count << "\t"
                          << acc.recall1 / n << "\t"
                          << acc.recall10 / n << "\t"
                          << acc.recall100 / n << "\t"
                          << acc.coverage10 / n << "\t"
                          << acc.coverage100 / n << "\t"
                          << acc.us / n << "\n";
            };
            print_row("FULL2BIT_NAV_FULL2BIT_RETURN", 100, full2bit_return);
            print_row("FULL2BIT_NAV_FLOAT32_RERANK_TOP100", 100, float_top100);
            print_row("FULL2BIT_NAV_FLOAT32_RERANK_TOP500", 500, float_top500);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "full2bit_navigation_oracle failed: " << e.what() << "\n";
        return 1;
    }
}
