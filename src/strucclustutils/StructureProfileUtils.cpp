#include "StructureProfileUtils.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "DBReader.h"
#include "DBWriter.h"
#include "BaseMatrix.h"
#include "SubstitutionMatrix.h"
#include "FileUtil.h"
#include "StructureUtil.h"
#include "structureto12st.h"

namespace {

std::string getSerializedMatrix(LocalParameters &par,
                                const std::string &matrixName,
                                const std::string &serializedName = "") {
    for (size_t i = 0; i < par.substitutionMatrices.size(); i++) {
        if (par.substitutionMatrices[i].name == matrixName) {
            std::string matrixData(
                reinterpret_cast<const char *>(par.substitutionMatrices[i].subMatData),
                par.substitutionMatrices[i].subMatDataLen
            );
            std::string mutableName = serializedName.empty() ? matrixName : serializedName;
            char *serializedMatrix = BaseMatrix::serialize(mutableName, matrixData);
            std::string result(serializedMatrix);
            free(serializedMatrix);
            return result;
        }
    }
    return "";
}

std::string createProfileParameters(LocalParameters &par,
                                    const std::string &matrixFile,
                                    const MultiParam<PseudoCounts> &pca,
                                    const MultiParam<PseudoCounts> &pcb,
                                    int compBiasCorrection,
                                    float compBiasCorrectionScale,
                                    int maskProfile,
                                    float maskProb,
                                    double evalProfile,
                                    bool forceEvalProfile,
                                    bool syncEvalThrWhenUnset,
                                    const MultiParam<NuclAA<int> > *gapOpenOverride = NULL,
                                    const MultiParam<NuclAA<int> > *gapExtendOverride = NULL) {
    const std::string savedScoringMatrixFile = par.scoringMatrixFile.values.aminoacid();
    const std::string savedSeedScoringMatrixFile = par.seedScoringMatrixFile.values.aminoacid();
    const MultiParam<PseudoCounts> savedPca = par.pca;
    const MultiParam<PseudoCounts> savedPcb = par.pcb;
    const int savedCompBiasCorrection = par.compBiasCorrection;
    const float savedCompBiasCorrectionScale = par.compBiasCorrectionScale;
    const int savedMaskProfile = par.maskProfile;
    const float savedMaskProb = par.maskProb;
    const double savedEvalProfile = par.evalProfile;
    const double savedEvalThr = par.evalThr;
    const MultiParam<NuclAA<int> > savedGapOpen = par.gapOpen;
    const MultiParam<NuclAA<int> > savedGapExtend = par.gapExtend;

    par.scoringMatrixFile = matrixFile;
    par.seedScoringMatrixFile = matrixFile;
    par.pca = pca;
    par.pcb = pcb;
    par.compBiasCorrection = compBiasCorrection;
    par.compBiasCorrectionScale = compBiasCorrectionScale;
    par.maskProfile = maskProfile;
    par.maskProb = maskProb;
    if (forceEvalProfile) {
        par.evalProfile = evalProfile;
    } else if (syncEvalThrWhenUnset && par.PARAM_E_PROFILE.wasSet == false) {
        par.evalProfile = evalProfile;
        par.evalThr = evalProfile;
    }
    if (gapOpenOverride != NULL) {
        par.gapOpen = *gapOpenOverride;
    }
    if (gapExtendOverride != NULL) {
        par.gapExtend = *gapExtendOverride;
    }

    const std::string profilePar = par.createParameterString(par.result2structprofile);

    par.scoringMatrixFile = savedScoringMatrixFile;
    par.seedScoringMatrixFile = savedSeedScoringMatrixFile;
    par.pca = savedPca;
    par.pcb = savedPcb;
    par.compBiasCorrection = savedCompBiasCorrection;
    par.compBiasCorrectionScale = savedCompBiasCorrectionScale;
    par.maskProfile = savedMaskProfile;
    par.maskProb = savedMaskProb;
    par.evalProfile = savedEvalProfile;
    par.evalThr = savedEvalThr;
    par.gapOpen = savedGapOpen;
    par.gapExtend = savedGapExtend;

    return profilePar;
}

void split3Di12StDb(const std::string &inputDb,
                    const std::string &out3DiDb,
                    const std::string &out12StDb,
                    LocalParameters &par) {
    DBReader<unsigned int> reader(inputDb.c_str(), (inputDb + ".index").c_str(), 1,
                                  DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    reader.open(DBReader<unsigned int>::NOSORT);

    std::string mat3Di = getSerializedMatrix(par, "3di.out");
    std::string mat12St = getSerializedMatrix(par, "12st.out");
    if (mat3Di.empty() || mat12St.empty()) {
        Debug(Debug::ERROR) << "Cannot find 3di.out or 12st.out substitution matrix\n";
        EXIT(EXIT_FAILURE);
    }
    SubstitutionMatrix subMat3Di(mat3Di.c_str(), 2.0, 0.0);
    SubstitutionMatrix subMat12St(mat12St.c_str(), par.submat12stScale, 0.0);

    DBWriter out3Di(out3DiDb.c_str(), (out3DiDb + ".index").c_str(), 1, false, Parameters::DBTYPE_AMINO_ACIDS);
    DBWriter out12St(out12StDb.c_str(), (out12StDb + ".index").c_str(), 1, false, Parameters::DBTYPE_AMINO_ACIDS);
    out3Di.open();
    out12St.open();

    std::vector<char> seq3Di;
    std::vector<char> seq12St;
    for (size_t id = 0; id < reader.getSize(); id++) {
        size_t seqLen = reader.getSeqLen(id);
        char *data = reader.getData(id, 0);
        StructureUtil::split3Di12St(data, seqLen, seq3Di, seq12St, subMat3Di, subMat12St, false);
        seq3Di.resize(seqLen + 1);
        seq12St.resize(seqLen + 1);
        seq3Di[seqLen] = '\n';
        seq12St[seqLen] = '\n';
        unsigned int dbKey = reader.getDbKey(id);
        out3Di.writeData(seq3Di.data(), seqLen + 1, dbKey, 0);
        out12St.writeData(seq12St.data(), seqLen + 1, dbKey, 0);
    }

    out3Di.close(true);
    out12St.close(true);
    reader.close();
}

} // namespace

ResolvedStructureProfileDbs resolveStructureProfileDbs(LocalParameters &par,
                                                       const std::string &db2NoIndexName,
                                                       bool disableSs12Profile) {
    ResolvedStructureProfileDbs resolved;
    resolved.ssInputDb = par.db1 + "_ss";
    resolved.ssTargetDb = db2NoIndexName + "_ss";
    resolved.removeSplitTmp = false;

    const bool packedInput3Di12St = FileUtil::fileExists((resolved.ssInputDb + ".dbtype").c_str())
        && StructureUtil::is3Di12StDb(FileUtil::parseDbType(resolved.ssInputDb.c_str()));
    const bool packedTarget3Di12St = FileUtil::fileExists((resolved.ssTargetDb + ".dbtype").c_str())
        && StructureUtil::is3Di12StDb(FileUtil::parseDbType(resolved.ssTargetDb.c_str()));

    if (packedInput3Di12St || packedTarget3Di12St) {
        if (db2NoIndexName != par.db2) {
            Debug(Debug::ERROR) << "12-state profile generation for indexed targets is not supported yet.\n";
            EXIT(EXIT_FAILURE);
        }

        if (packedInput3Di12St) {
            resolved.ssInputDb = par.db4 + "_split_query_ss3di";
            resolved.ss12InputDb = par.db4 + "_split_query_ss12";
            split3Di12StDb(par.db1 + "_ss", resolved.ssInputDb, resolved.ss12InputDb, par);
            resolved.removeSplitTmp = true;
        }

        if (packedTarget3Di12St) {
            resolved.ssTargetDb = par.db4 + "_split_target_ss3di";
            resolved.ss12TargetDb = par.db4 + "_split_target_ss12";
            split3Di12StDb(db2NoIndexName + "_ss", resolved.ssTargetDb, resolved.ss12TargetDb, par);
            resolved.removeSplitTmp = true;
        }
    }

    // A profile query (e.g. iterations >= 2 of an iterative 12-state search) has an HMM-profile
    // _ss DB that cannot be split. If the previous iteration left a companion _ss12 profile next
    // to it, reuse it directly so the 12-state channel is carried forward instead of being
    // dropped -- otherwise the next --evalue-nn-mode 2 alignment aborts with
    // "requires a query 12-state alphabet in the packed _ss DB or a _ss12 DB".
    if (packedInput3Di12St == false
        && resolved.ss12InputDb.empty()
        && FileUtil::fileExists((par.db1 + "_ss12.dbtype").c_str())) {
        resolved.ss12InputDb = par.db1 + "_ss12";
    }

    if (disableSs12Profile) {
        resolved.ss12InputDb.clear();
        resolved.ss12TargetDb.clear();
    }

    return resolved;
}

std::string createAaProfileParameters(LocalParameters &par) {
    return createProfileParameters(par,
                                   "blosum62.out",
                                   MultiParam<PseudoCounts>(PseudoCounts(1.1, 1.4)),
                                   MultiParam<PseudoCounts>(PseudoCounts(4.1, 5.8)),
                                   1,
                                   1.0f,
                                   1,
                                   0.9f,
                                   0.001,
                                   true,
                                   false);
}

std::string create3DiProfileParameters(LocalParameters &par) {
    return createProfileParameters(par,
                                   "3di.out",
                                   MultiParam<PseudoCounts>(PseudoCounts(1.4, 1.4)),
                                   MultiParam<PseudoCounts>(PseudoCounts(1.5, 1.5)),
                                   0,
                                   1.0f,
                                   0,
                                   0.9f,
                                   0.1,
                                   false,
                                   true);
}

std::string create12StateProfileParameters(LocalParameters &par) {
    const std::string matrixAlias = getSerializedMatrix(par, "12st.out", "blosum62.out");
    const MultiParam<NuclAA<int> > gapOpen(NuclAA<int>(11, par.gapOpen.values.nucleotide()));
    const MultiParam<NuclAA<int> > gapExtend(NuclAA<int>(1, par.gapExtend.values.nucleotide()));
    return createProfileParameters(par,
                                   matrixAlias,
                                   MultiParam<PseudoCounts>(PseudoCounts(1.4, 1.4)),
                                   MultiParam<PseudoCounts>(PseudoCounts(1.5, 1.5)),
                                   0,
                                   1.0f,
                                   0,
                                   0.9f,
                                   0.1,
                                   false,
                                   true,
                                   &gapOpen,
                                   &gapExtend);
}
