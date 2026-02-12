//
// Created by Martin Steinegger on 19/01/2022.
//

#include "EvalueNeuralNet.h"
#include "evalue_nn.kerasify.h"


EvalueNeuralNet::EvalueNeuralNet(size_t dbResCount, BaseMatrix* subMat) : subMat(subMat) {
        logDbResidueCount = log(static_cast<double>(dbResCount));
        encoder.LoadModel(
        std::string((const char *)evalue_nn_kerasify,
        evalue_nn_kerasify_len));
        in = Tensor(subMat->alphabetSize + 1);
        out = Tensor(2);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(unsigned char *, unsigned int L){
    if (L < 1) {
        return std::make_pair(0.188743, -2.412505); // global means as fallback
    }
    const double logL = std::log(static_cast<double>(L));
    // μ(L) = a + b log(L)
    const double mu =
        -0.723212
        - 0.321864 * logL;
    // λ(L) = c + d / sqrt(L)
    const double lambda =
        0.118086
        + 0.889118 / std::sqrt(static_cast<double>(L));
    return std::make_pair(lambda, mu);
}
