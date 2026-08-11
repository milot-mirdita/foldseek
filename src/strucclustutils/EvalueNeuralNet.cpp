//
// Created by Martin Steinegger on 19/01/2022.
//

#include "EvalueNeuralNet.h"
#include "Sequence.h"
#include "evalue_nn.kerasify.h"
#include "evalue_12st_nn.kerasify.h"
#include "windowed_evalue_nn.kerasify.h"
#include <algorithm>
#include <numeric>

namespace {
static const char WINDOWED_EVALUE_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
};
static const int WINDOWED_EVALUE_ALPHABET_SIZE = sizeof(WINDOWED_EVALUE_ALPHABET) / sizeof(WINDOWED_EVALUE_ALPHABET[0]);
static const char LEGACY_12ST_THREEDI_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
};
static const char LEGACY_12ST_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'X'
};
static const int LEGACY_12ST_THREEDI_ALPHABET_SIZE = sizeof(LEGACY_12ST_THREEDI_ALPHABET) / sizeof(LEGACY_12ST_THREEDI_ALPHABET[0]);
static const int LEGACY_12ST_ALPHABET_SIZE = sizeof(LEGACY_12ST_ALPHABET) / sizeof(LEGACY_12ST_ALPHABET[0]);
}


EvalueNeuralNet::EvalueNeuralNet(size_t dbResidueCount,
                                 size_t dbSequenceCount,
                                 BaseMatrix* subMat,
                                 int mode,
                                 bool,
                                 bool recoverProfileComposition)
                                 : subMat(subMat), mode(mode), recoverProfileComposition(recoverProfileComposition) {
        logDbResidueCount = log(static_cast<double>(dbResidueCount));
        logDbSequenceCount = log(static_cast<double>(dbSequenceCount));
        if (usesWindowedModel()) {
            stateToWindowedIndex.assign(subMat->alphabetSize, -1);
            windowedBackground.assign(WINDOWED_EVALUE_ALPHABET_SIZE, 0.0f);
            for (int i = 0; i < WINDOWED_EVALUE_ALPHABET_SIZE; ++i) {
                const unsigned char aa = static_cast<unsigned char>(WINDOWED_EVALUE_ALPHABET[i]);
                const unsigned char state = subMat->aa2num[aa];
                if (state < stateToWindowedIndex.size()) {
                    stateToWindowedIndex[state] = i;
                    windowedBackground[i] = static_cast<float>(subMat->pBack[state]);
                }
            }
            windowedEncoder.LoadModel(
                    std::string((const char *)windowed_evalue_nn_kerasify,
                                windowed_evalue_nn_kerasify_len));
            windowedIn = Tensor(42);
            windowedOut = Tensor(2);
        } else if (usesLegacy12StModel()) {
            encoder.LoadModel(
                    std::string((const char *)evalue_12st_nn_kerasify,
                                evalue_12st_nn_kerasify_len));
            in = Tensor(LEGACY_12ST_FEATURE_SIZE);
            out = Tensor(2);
        } else {
            encoder.LoadModel(
            std::string((const char *)evalue_nn_kerasify,
            evalue_nn_kerasify_len));
            in = Tensor(subMat->alphabetSize + 1);
            out = Tensor(2);
        }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const unsigned char * seq, unsigned int L){
    if (usesWindowedModel()) {
        return predictMuLambda(seq, L, static_cast<int>(L), seq, L, static_cast<int>(L), static_cast<double>(L));
    }
    for(int i = 0; i < subMat->alphabetSize; i++){
        in.data_[i] = 0;
    }
    for (unsigned int i = 0; i < L; i++) {
        in.data_[seq[i]]++;
    }
    in.data_[subMat->alphabetSize] = L;
    encoder.Apply(&in, &out);
    // used to normalize the output
    double mu1 = 0.17518475184751847;
    double sigma1 = 0.03260331312698818;
    double mu2 = -2.5569312493124934;
    double sigmal2 = 0.4353169278257701;
    return std::make_pair(out.data_[0]*sigma1+mu1,
                          out.data_[1]*sigmal2+mu2);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const Sequence &seq){
    if (usesWindowedModel()) {
        return predictMuLambda(seq, static_cast<int>(seq.L), seq, static_cast<int>(seq.L), static_cast<double>(seq.L));
    }
    return predictMuLambda(seq.numSequence, seq.L);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const Sequence &seq3Di, const Sequence &seq12St) {
    if (!usesLegacy12StModel()) {
        return predictMuLambda(seq3Di);
    }
    return predictMuLambdaLegacy12St(seq3Di, seq12St);
}

float EvalueNeuralNet::softplus(float x) {
    return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.0f);
}

int EvalueNeuralNet::predictedAlignmentLength(double score) const {
    const float pred = WINDOWED_PRED_SLOPE * static_cast<float>(score) + WINDOWED_PRED_INTERCEPT;
    return std::max(1, static_cast<int>(std::lround(pred)));
}

void EvalueNeuralNet::fillLegacy12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize) const {
    std::fill(dst, dst + alphabetSize, 0.0f);
    if (seq.L <= 0) {
        return;
    }
    // Profile query (e.g. iterations >= 2 of an iterative search): recover an expected
    // composition from the profile scores instead of counting the single center/query
    // sequence, mirroring the windowed model. Each column's profile scores are converted
    // back to probabilities, normalized to a per-position distribution, and averaged over
    // the profile. seq.subMat is used per channel (3Di or 12-state) so the background and
    // letter->state mapping match the sequence-path counting done below.
    if (recoverProfileComposition
        && Parameters::isEqualDbtype(seq.getSequenceType(), Parameters::DBTYPE_HMM_PROFILE)) {
        const short *profileScores = seq.profile_score;
        const size_t rowSize = seq.profile_row_size;
        std::vector<float> col(alphabetSize);
        for (int pos = 0; pos < seq.L; ++pos) {
            const short *row = profileScores + (static_cast<size_t>(pos) * rowSize);
            float colSum = 0.0f;
            for (int aa = 0; aa < alphabetSize; ++aa) {
                const unsigned char state = seq.subMat->aa2num[static_cast<unsigned char>(alphabet[aa])];
                float prob = 0.0f;
                // Only the PROFILE_AA_SIZE score columns are stored; states outside that range
                // (e.g. the neutral 'X') have no profile column and contribute nothing.
                if (state < static_cast<unsigned char>(Sequence::PROFILE_AA_SIZE)) {
                    prob = Sequence::scoreToProba(row[state], seq.subMat->pBack[state],
                                                  LEGACY_12ST_PROFILE_BIT_FACTOR, seq.subMat->scoreBias);
                }
                col[aa] = prob;
                colSum += prob;
            }
            if (colSum > 0.0f) {
                const float invColSum = 1.0f / colSum;
                for (int aa = 0; aa < alphabetSize; ++aa) {
                    dst[aa] += col[aa] * invColSum;
                }
            }
        }
        const float invLen = 1.0f / static_cast<float>(seq.L);
        for (int aa = 0; aa < alphabetSize; ++aa) {
            dst[aa] *= invLen;
        }
        return;
    }

    const float invLen = 1.0f / static_cast<float>(seq.L);
    for (int aa = 0; aa < alphabetSize; ++aa) {
        const unsigned char state = seq.subMat->aa2num[static_cast<unsigned char>(alphabet[aa])];
        for (int pos = 0; pos < seq.L; ++pos) {
            if (seq.numSequence[pos] == state) {
                dst[aa] += invLen;
            }
        }
    }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambdaLegacy12St(const Sequence &seq3Di, const Sequence &seq12St) {
    std::fill(in.data_.begin(), in.data_.end(), 0.0f);
    fillLegacy12StComposition(in.data_.data(), seq3Di, LEGACY_12ST_THREEDI_ALPHABET, LEGACY_12ST_THREEDI_ALPHABET_SIZE);
    fillLegacy12StComposition(in.data_.data() + LEGACY_12ST_THREEDI_ALPHABET_SIZE,
                              seq12St, LEGACY_12ST_ALPHABET, LEGACY_12ST_ALPHABET_SIZE);
    in.data_[LEGACY_12ST_FEATURE_SIZE - 1] = static_cast<float>(seq3Di.L);
    encoder.Apply(&in, &out);
    return std::make_pair(out.data_[0] * LEGACY_12ST_LAMBDA_STD + LEGACY_12ST_LAMBDA_MEAN,
                          out.data_[1] * LEGACY_12ST_MU_STD + LEGACY_12ST_MU_MEAN);
}

void EvalueNeuralNet::fillWindowedComposition(float *dst, const unsigned char *seq, unsigned int seqLen, int end1Based, int spanLen) const {
    std::fill(dst, dst + WINDOWED_EVALUE_ALPHABET_SIZE, 0.0f);
    const int end = std::max(0, std::min(static_cast<int>(seqLen), end1Based));
    const int start = std::max(0, end - spanLen);
    const int span = end - start;
    if (span <= 0) {
        return;
    }
    const float invLen = 1.0f / static_cast<float>(span);
    for (int i = start; i < end; ++i) {
        const unsigned char state = seq[i];
        if (state < stateToWindowedIndex.size()) {
            const int featIdx = stateToWindowedIndex[state];
            if (featIdx >= 0) {
                dst[featIdx] += invLen;
            }
        }
    }
}

void EvalueNeuralNet::fillWindowedComposition(float *dst, const Sequence &seq, int end1Based, int spanLen) const {
    if (!Parameters::isEqualDbtype(seq.getSequenceType(), Parameters::DBTYPE_HMM_PROFILE)) {
        fillWindowedComposition(dst, seq.numSequence, seq.L, end1Based, spanLen);
        return;
    }

    std::fill(dst, dst + WINDOWED_EVALUE_ALPHABET_SIZE, 0.0f);
    const int end = std::max(0, std::min(seq.L, end1Based));
    const int start = std::max(0, end - spanLen);
    const int span = end - start;
    if (span <= 0) {
        return;
    }

    const short *profileScores = seq.profile_score;
    const size_t rowSize = seq.profile_row_size;
    for (int pos = start; pos < end; ++pos) {
        const short *row = profileScores + (static_cast<size_t>(pos) * rowSize);
        float col[WINDOWED_EVALUE_ALPHABET_SIZE];
        float colSum = 0.0f;
        for (int aa = 0; aa < WINDOWED_EVALUE_ALPHABET_SIZE; ++aa) {
            const unsigned char state = subMat->aa2num[static_cast<unsigned char>(WINDOWED_EVALUE_ALPHABET[aa])];
            const float prob = (state < subMat->alphabetSize)
                ? Sequence::scoreToProba(row[aa], subMat->pBack[state], WINDOWED_PROFILE_BIT_FACTOR, subMat->scoreBias)
                : 0.0f;
            col[aa] = prob;
            colSum += prob;
        }
        if (colSum > 0.0f) {
            const float invColSum = 1.0f / colSum;
            for (int aa = 0; aa < WINDOWED_EVALUE_ALPHABET_SIZE; ++aa) {
                dst[aa] += col[aa] * invColSum;
            }
        } else if (!windowedBackground.empty()) {
            for (int aa = 0; aa < WINDOWED_EVALUE_ALPHABET_SIZE; ++aa) {
                dst[aa] += windowedBackground[aa];
            }
        }
    }

    const float invSpan = 1.0f / static_cast<float>(span);
    for (int aa = 0; aa < WINDOWED_EVALUE_ALPHABET_SIZE; ++aa) {
        dst[aa] *= invSpan;
    }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambdaWindowed(const unsigned char *querySeq,
                                                                   unsigned int queryLen,
                                                                   int queryEnd1Based,
                                                                   const unsigned char *targetSeq,
                                                                   unsigned int targetLen,
                                                                   int targetEnd1Based,
                                                                   double score) {
    std::fill(windowedIn.data_.begin(), windowedIn.data_.end(), 0.0f);
    const int predLen = predictedAlignmentLength(score);
    windowedIn.data_[0] = static_cast<float>(queryLen);
    windowedIn.data_[1] = static_cast<float>(targetLen);
    fillWindowedComposition(windowedIn.data_.data() + 2, querySeq, queryLen, queryEnd1Based, predLen);
    fillWindowedComposition(windowedIn.data_.data() + 2 + WINDOWED_EVALUE_ALPHABET_SIZE,
                            targetSeq, targetLen, targetEnd1Based, predLen);
    windowedEncoder.Apply(&windowedIn, &windowedOut);
    const double outMu = static_cast<double>(windowedOut.data_[0]);
    const double outLambdaRaw = static_cast<double>(windowedOut.data_[1]);
    const double lambda = static_cast<double>(softplus(static_cast<float>(outLambdaRaw)) + WINDOWED_MIN_LAMBDA);
    return std::make_pair(lambda, outMu);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambdaWindowed(const Sequence &querySeq,
                                                                   int queryEnd1Based,
                                                                   const Sequence &targetSeq,
                                                                   int targetEnd1Based,
                                                                   double score) {
    std::fill(windowedIn.data_.begin(), windowedIn.data_.end(), 0.0f);
    const int predLen = predictedAlignmentLength(score);
    windowedIn.data_[0] = static_cast<float>(querySeq.L);
    windowedIn.data_[1] = static_cast<float>(targetSeq.L);
    fillWindowedComposition(windowedIn.data_.data() + 2, querySeq, queryEnd1Based, predLen);
    fillWindowedComposition(windowedIn.data_.data() + 2 + WINDOWED_EVALUE_ALPHABET_SIZE,
                            targetSeq, targetEnd1Based, predLen);
    windowedEncoder.Apply(&windowedIn, &windowedOut);
    const double outMu = static_cast<double>(windowedOut.data_[0]);
    const double outLambdaRaw = static_cast<double>(windowedOut.data_[1]);
    const double lambda = static_cast<double>(softplus(static_cast<float>(outLambdaRaw)) + WINDOWED_MIN_LAMBDA);
    return std::make_pair(lambda, outMu);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const unsigned char *querySeq,
                                                           unsigned int queryLen,
                                                           int queryEnd1Based,
                                                           const unsigned char *targetSeq,
                                                           unsigned int targetLen,
                                                           int targetEnd1Based,
                                                           double score) {
    if (!usesWindowedModel()) {
        return predictMuLambda(querySeq, queryLen);
    }
    return predictMuLambdaWindowed(querySeq, queryLen, queryEnd1Based, targetSeq, targetLen, targetEnd1Based, score);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const Sequence &querySeq,
                                                           int queryEnd1Based,
                                                           const Sequence &targetSeq,
                                                           int targetEnd1Based,
                                                           double score) {
    if (!usesWindowedModel()) {
        return predictMuLambda(querySeq);
    }
    return predictMuLambdaWindowed(querySeq, queryEnd1Based, targetSeq, targetEnd1Based, score);
}
