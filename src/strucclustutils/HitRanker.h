#ifndef FOLDSEEK_HITRANKER_H
#define FOLDSEEK_HITRANKER_H

// Re-ranks the hits of a query with a gradient-boosted tree model (LightGBM text format).
// Features use the alignment and the 3Di/AA sequences, optionally the target C-alpha coordinates
// and the lDDT and TM-score of the alignment.

#include "Matcher.h"

#include <string>
#include <vector>

class BaseMatrix;

namespace HitRanker {

// names of the values written by computeFeatures, in this order
extern const std::vector<std::string> FEATURE_NAMES;

struct Chain {
    const unsigned char *seqAA;
    const unsigned char *seq3Di;
    int length;
    const float *x, *y, *z;     // C-alpha coordinates, NULL if not available

    // secondary structure elements derived from 3Di
    std::vector<int> element;   // element of each residue, -1 for coil
    std::vector<char> type;     // 'H' or 'E' per element
    std::vector<int> start;
    std::vector<int> end;       // exclusive
    void setSecondaryStructure(const BaseMatrix &subMat3Di);
};

// lddt and tmscore are NAN when the query has no coordinates
void computeFeatures(const Chain &query, const Chain &target, const Matcher::result_t &res,
                     const BaseMatrix &subMatAA, const BaseMatrix &subMat3Di,
                     float lddt, float tmscore, float *features);

class Model {
public:
    // built-in model: with both C-alpha (lDDT/TM features), target C-alpha only (contacts) or none
    void loadBuiltin(bool queryHasCalpha, bool targetHasCalpha);
    // LightGBM text model; name is used in error messages
    void load(const std::string &text, const std::string &name);
    // features of hit h are at columns[f * nHits + h]
    void predict(const float *columns, size_t nHits, double *scores) const;

private:
    struct Node {
        float threshold;    // largest float <= the LightGBM threshold
        int feature;
        int child[2];       // [0] if value <= threshold; leaves point to themselves
    };
    std::vector<Node> nodes;
    std::vector<double> leafValue;  // per node
    std::vector<int> roots;
    std::vector<int> used;          // feature index of each model feature
};

}

#endif
