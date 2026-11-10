#include "HitRanker.h"
#include "BaseMatrix.h"
#include "Debug.h"
#include "Util.h"
#include "hitrank_native.txt.h"
#include "hitrank_target.txt.h"
#include "hitrank_seq.txt.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace HitRanker {

const std::vector<std::string> FEATURE_NAMES = {
    "bits", "logE", "alnlen", "fident", "gapopen", "gapres", "bits_per_col", "nM",
    "s3_sum", "sa_sum", "s3_mean", "sa_mean", "id3", "pos3", "posrun", "qspan", "tspan", "span_ratio",
    "cont_both", "cont_span", "rg_ratio",
    "a_col_same", "a_col_wrong", "a_col_qsse_gap", "a_col_tsse_gap",
    "a_match_q", "a_match_t", "a_unm_qE", "a_unm_qH", "a_unm_tE", "a_unm_tH",
    "native", "lddt", "tm"
};

void Chain::setSecondaryStructure(const BaseMatrix &subMat3Di) {
    // 0 = H, 1 = E, 2 = coil
    std::vector<unsigned char> raw(length);
    for (int i = 0; i < length; i++) {
        switch (subMat3Di.num2aa[seq3Di[i]]) {
            case 'E': case 'I': case 'K': case 'M': case 'T': case 'W': case 'Y': raw[i] = 1; break;
            case 'C': case 'L': case 'N': case 'S': case 'V': raw[i] = 0; break;
            default: raw[i] = 2;
        }
    }
    // majority over i-1..i+1; ties go to the residue's own state, then to H, E, coil
    std::vector<unsigned char> state(length);
    for (int i = 0; i < length; i++) {
        int votes[3] = {0, 0, 0};
        votes[raw[i]] += 3;
        if (i > 0) votes[raw[i - 1]] += 2;
        if (i + 1 < length) votes[raw[i + 1]] += 2;
        int best = 0;
        if (votes[1] > votes[best]) best = 1;
        if (votes[2] > votes[best]) best = 2;
        state[i] = best;
    }
    element.assign(length, -1);
    type.clear();
    start.clear();
    end.clear();
    for (int i = 0, j; i < length; i = j) {
        for (j = i; j < length && state[j] == state[i]; j++) {}
        if (state[i] != 2 && j - i >= (state[i] == 0 ? 4 : 3)) {
            std::fill(element.begin() + i, element.begin() + j, static_cast<int>(type.size()));
            type.push_back(state[i] == 0 ? 'H' : 'E');
            start.push_back(i);
            end.push_back(j);
        }
    }
}

// for one side: elements inside [from, to], how many are matched, and the unmatched ones by type
static void elementAgreement(const Chain &c, const std::vector<int> &sameCount, int from, int to,
                             float &matchedFraction, float &unmatchedE, float &unmatchedH) {
    int inside = 0, matched = 0;
    unmatchedE = unmatchedH = 0;
    for (size_t e = 0; e < c.type.size(); e++) {
        const int len = c.end[e] - c.start[e];
        if (2 * (std::min(c.end[e], to + 1) - std::max(c.start[e], from)) < len) {
            continue;
        }
        inside++;
        if (2 * sameCount[e] >= len) {
            matched++;
        } else if (c.type[e] == 'E') {
            unmatchedE++;
        } else {
            unmatchedH++;
        }
    }
    matchedFraction = static_cast<float>(matched) / std::max(1, inside);
}

void computeFeatures(const Chain &query, const Chain &target, const Matcher::result_t &res,
                     const BaseMatrix &subMatAA, const BaseMatrix &subMat3Di,
                     float lddt, float tmscore, float *f) {
    std::fill(f, f + FEATURE_NAMES.size(), NAN);
    std::vector<int> qi, ti;
    int gapOpen = 0, gapResidues = 0;
    char previous = 'M';
    for (int i = res.qStartPos, j = res.dbStartPos, k = 0; k < static_cast<int>(res.backtrace.size()); k++) {
        const char c = res.backtrace[k];
        if (c == 'M') {
            qi.push_back(i++);
            ti.push_back(j++);
        } else {
            gapOpen += (c != previous);
            gapResidues++;
            (c == 'I') ? i++ : j++;
        }
        previous = c;
    }
    const int nM = static_cast<int>(qi.size());
    const float alnLen = static_cast<float>(res.alnLength);
    const float cols = static_cast<float>(std::max(1, nM));

    float sum3Di = 0, sumAA = 0;
    int identical = 0, positive = 0, run = 0, bestRun = 0;
    int same = 0, wrong = 0, qInElement = 0, tInElement = 0, qOnly = 0, tOnly = 0;
    std::vector<int> qSame(query.type.size(), 0), tSame(target.type.size(), 0);
    for (int k = 0; k < nM; k++) {
        const unsigned char q3 = query.seq3Di[qi[k]], t3 = target.seq3Di[ti[k]];
        const short s3 = subMat3Di.subMatrix[q3][t3];
        sum3Di += s3;
        sumAA += subMatAA.subMatrix[query.seqAA[qi[k]]][target.seqAA[ti[k]]];
        identical += (q3 == t3);
        positive += (s3 > 0);
        run = (s3 > 0) ? run + 1 : 0;
        bestRun = std::max(bestRun, run);

        const int qe = query.element[qi[k]], te = target.element[ti[k]];
        qInElement += (qe >= 0);
        tInElement += (te >= 0);
        qOnly += (qe >= 0 && te < 0);
        tOnly += (te >= 0 && qe < 0);
        if (qe >= 0 && te >= 0) {
            if (query.type[qe] == target.type[te]) {
                same++;
                qSame[qe]++;
                tSame[te]++;
            } else {
                wrong++;
            }
        }
    }
    const float qSpan = static_cast<float>(res.qEndPos - res.qStartPos + 1);
    const float tSpan = static_cast<float>(res.dbEndPos - res.dbStartPos + 1);
    f[0] = static_cast<float>(res.score);
    f[1] = static_cast<float>(log10(std::max(res.eval, 1e-300)));
    f[2] = alnLen;
    f[3] = res.seqId;
    f[4] = static_cast<float>(gapOpen);
    f[5] = gapResidues / alnLen;
    f[6] = res.score / alnLen;
    f[7] = static_cast<float>(nM);
    f[8] = sum3Di;
    f[9] = sumAA;
    f[10] = sum3Di / cols;
    f[11] = sumAA / cols;
    f[12] = identical / cols;
    f[13] = positive / cols;
    f[14] = bestRun / alnLen;
    f[15] = qSpan;
    f[16] = tSpan;
    f[17] = std::min(qSpan, tSpan) / std::max(qSpan, tSpan);

    // target contacts: C-alpha pairs closer than 10 A, at least 3 residues apart
    if (target.x != NULL) {
        std::vector<int> aligned(target.length, 0), inSpan(target.length, 0);
        for (int t : ti) aligned[t] = 1;
        std::fill(inSpan.begin() + res.dbStartPos, inSpan.begin() + res.dbEndPos + 1, 1);
        long incident = 0, both = 0, spanPairs = 0, spanBoth = 0;
        for (int a = 0; a < target.length; a++) {
            const float ax = target.x[a], ay = target.y[a], az = target.z[a];
            const int alignedA = aligned[a], inSpanA = inSpan[a];
            int rowIncident = 0, rowBoth = 0, rowSpan = 0, rowSpanBoth = 0;
            // branch free so that the compiler can vectorize it
            for (int b = a + 3; b < target.length; b++) {
                const float dx = ax - target.x[b], dy = ay - target.y[b], dz = az - target.z[b];
                const int contact = (dx * dx + dy * dy + dz * dz) < 10.0f * 10.0f;
                const int pairAligned = alignedA & aligned[b];
                const int pairInSpan = contact & inSpanA & inSpan[b];
                rowIncident += contact & (alignedA | aligned[b]);
                rowBoth += contact & pairAligned;
                rowSpan += pairInSpan;
                rowSpanBoth += pairInSpan & pairAligned;
            }
            incident += rowIncident;
            both += rowBoth;
            spanPairs += rowSpan;
            spanBoth += rowSpanBoth;
        }
        double cx = 0, cy = 0, cz = 0, sq = 0;
        for (int t : ti) {
            cx += target.x[t];
            cy += target.y[t];
            cz += target.z[t];
        }
        cx /= std::max(1, nM);
        cy /= std::max(1, nM);
        cz /= std::max(1, nM);
        for (int t : ti) {
            sq += (target.x[t] - cx) * (target.x[t] - cx) + (target.y[t] - cy) * (target.y[t] - cy)
                + (target.z[t] - cz) * (target.z[t] - cz);
        }
        const double rg = sqrt(sq / std::max(1, nM));
        f[18] = static_cast<float>(both) / std::max(1L, incident);
        f[19] = static_cast<float>(spanBoth) / std::max(1L, spanPairs);
        f[20] = static_cast<float>(rg / (2.2 * pow(std::max(1, nM), 0.38)));
    }

    f[21] = same / cols;
    f[22] = wrong / cols;
    f[23] = static_cast<float>(qOnly) / std::max(1, qInElement);
    f[24] = static_cast<float>(tOnly) / std::max(1, tInElement);
    elementAgreement(query, qSame, res.qStartPos, res.qEndPos, f[25], f[27], f[28]);
    elementAgreement(target, tSame, res.dbStartPos, res.dbEndPos, f[26], f[29], f[30]);

    if (std::isnan(lddt) == false) {
        const float tm = std::min(1.0f, tmscore);
        f[31] = sqrt(std::max(0.0f, lddt) * tm);
        f[32] = lddt;
        f[33] = tm;
    }
}

static void fail(const std::string &name, const std::string &reason) {
    Debug(Debug::ERROR) << "Rank model " << name << ": " << reason << "\n";
    EXIT(EXIT_FAILURE);
}

void Model::loadBuiltin(bool queryHasCalpha, bool targetHasCalpha) {
    if (queryHasCalpha && targetHasCalpha) {
        load(std::string((const char *)hitrank_native_txt, hitrank_native_txt_len), "hitrank_native");
    } else if (targetHasCalpha) {
        load(std::string((const char *)hitrank_target_txt, hitrank_target_txt_len), "hitrank_target");
    } else {
        load(std::string((const char *)hitrank_seq_txt, hitrank_seq_txt_len), "hitrank_seq");
    }
}

void Model::load(const std::string &text, const std::string &path) {
    std::istringstream in(text);
    const std::vector<std::string> keys = {"split_feature", "threshold", "decision_type", "left_child", "right_child", "leaf_value"};
    std::vector<std::vector<double>> tree;  // values of the keys above for the current tree
    auto addTree = [&]() {
        if (tree.empty()) return;
        const int offset = static_cast<int>(nodes.size());
        const int splits = static_cast<int>(tree[0].size());
        // children >= 0 are splits, < 0 are ~leaf; leaves are stored after the splits
        auto child = [&](double c) { return offset + (c >= 0 ? static_cast<int>(c) : splits + ~static_cast<int>(c)); };
        for (int n = 0; n < splits; n++) {
            if (tree[2][n] != 0 && tree[2][n] != 2) {
                fail(path, "only numerical splits without missing values are supported");
            }
            float threshold = static_cast<float>(tree[1][n]);
            if (threshold > tree[1][n]) {
                threshold = std::nextafter(threshold, -INFINITY);
            }
            nodes.push_back({threshold, used.at(static_cast<int>(tree[0][n])), {child(tree[3][n]), child(tree[4][n])}});
            leafValue.push_back(0.0);
        }
        for (double value : tree[5]) {
            const int self = static_cast<int>(nodes.size());
            nodes.push_back({INFINITY, 0, {self, self}});
            leafValue.push_back(value);
        }
        roots.push_back(offset);
        tree.clear();
    };
    std::string line;
    while (std::getline(in, line) && line != "end of trees") {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        std::istringstream values(line.substr(eq + 1));
        if (key == "feature_names") {
            for (std::string name; values >> name;) {
                auto it = std::find(FEATURE_NAMES.begin(), FEATURE_NAMES.end(), name);
                if (it == FEATURE_NAMES.end()) fail(path, "unknown feature " + name);
                used.push_back(static_cast<int>(it - FEATURE_NAMES.begin()));
            }
        } else if (key == "Tree") {
            addTree();
            tree.assign(keys.size(), std::vector<double>());
        } else if ((key == "num_cat" || key == "is_linear") && line.substr(eq + 1) != "0") {
            fail(path, "categorical splits and linear trees are not supported");
        } else if (tree.empty() == false) {
            auto it = std::find(keys.begin(), keys.end(), key);
            for (double v; it != keys.end() && values >> v;) tree[it - keys.begin()].push_back(v);
        }
    }
    addTree();
    if (roots.empty()) fail(path, "no trees");
}

void Model::predict(const float *columns, size_t nHits, double *scores) const {
    std::fill(scores, scores + nHits, 0.0);
    // tree by tree, so that a tree stays in cache for all hits; the HITS walks overlap
    const size_t HITS = 8;
    for (int root : roots) {
        for (size_t first = 0; first < nHits; first += HITS) {
            const size_t count = std::min(HITS, nHits - first);
            const float *lane[HITS];
            int node[HITS];
            for (size_t k = 0; k < HITS; k++) {
                lane[k] = columns + first + std::min(k, count - 1);  // a short block repeats its last hit
                node[k] = root;
            }
            for (bool moved = true; moved;) {
                moved = false;
                for (size_t k = 0; k < HITS; k++) {
                    const Node &n = nodes[node[k]];
                    // index instead of branch: the comparison is unpredictable
                    const int next = n.child[!(lane[k][n.feature * nHits] <= n.threshold)];
                    moved |= (next != node[k]);
                    node[k] = next;
                }
            }
            for (size_t k = 0; k < count; k++) {
                scores[first + k] += leafValue[node[k]];
            }
        }
    }
}

}
