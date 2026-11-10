#include "DBReader.h"
#include "IndexReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "LocalParameters.h"
#include "Matcher.h"
#include "Alignment.h"
#include "structureto3diseqdist.h"
#include "StructureSmithWaterman.h"
#include "StructureUtil.h"
#include "TMaligner.h"
#include "Coordinate16.h"
#include "LDDT.h"
#include "structureto12st.h"
#include "HitRanker.h"


#ifdef OPENMP
#include <omp.h>
#endif
#include <memory>

// need for sorting the     results
static bool compareHitsByStructureBits(const Matcher::result_t &first, const Matcher::result_t &second) {
    if (first.score != second.score) {
        return first.score > second.score;
    }
    if (first.dbLen != second.dbLen) {
        return first.dbLen < second.dbLen;
    }
    return first.dbKey < second.dbKey;
}


static void structureAlignDefault(LocalParameters & par) {
    par.compBiasCorrectionScale = 0.5;
    par.alignmentType = LocalParameters::ALIGNMENT_TYPE_3DI_AA;
}

int alignStructure(StructureSmithWaterman & structureSmithWaterman,
                   StructureSmithWaterman & reverseStructureSmithWaterman,
                   Sequence & tSeqAA, Sequence & tSeq3Di,
                   Sequence * tSeq12St,
                   unsigned int querySeqLen, unsigned int targetSeqLen,
                   EvalueNeuralNet & evaluer, std::pair<double, double> queryMuLambda,
                   const std::pair<double, double> *targetMuLambda,
                   Matcher::result_t & res, std::string & backtrace,
                   Parameters & par, bool useReverseScore) {

    float seqId = 0.0;
    backtrace.clear();
    const unsigned char *db12StSeq = (tSeq12St != NULL) ? tSeq12St->numSequence : NULL;
    // align only score and end pos
    StructureSmithWaterman::s_align align = structureSmithWaterman.alignScoreEndPos<StructureSmithWaterman::PROFILE>(tSeqAA.numSequence, tSeq3Di.numSequence, targetSeqLen, par.gapOpen.values.aminoacid(),
                                                                                    par.gapExtend.values.aminoacid(), querySeqLen / 2, db12StSeq);
    bool hasLowerCoverage = !(Util::hasCoverage(par.covThr, par.covMode, align.qCov, align.tCov));
    if(hasLowerCoverage){
        return -1;
    }
    // we can already stop if this e-value isn't good enough, it wont be any better in the next step
    align.evalue = evaluer.computeEvalueCorr(align.score1, queryMuLambda.first, queryMuLambda.second, querySeqLen, targetSeqLen, targetMuLambda);
    bool hasLowerEvalue = align.evalue > par.evalThr;
    if(hasLowerEvalue){
        return -1;
    }

    const int32_t rawSmithWatermanScore = static_cast<int32_t>(align.score1);
    int32_t score;
    if (useReverseScore) {
        StructureSmithWaterman::s_align revAlign;
        revAlign = reverseStructureSmithWaterman.alignScoreEndPos<StructureSmithWaterman::PROFILE>(tSeqAA.numSequence, tSeq3Di.numSequence,
                                                                      targetSeqLen, par.gapOpen.values.aminoacid(),
                                                                      par.gapExtend.values.aminoacid(), querySeqLen / 2, db12StSeq);
        score = rawSmithWatermanScore - static_cast<int32_t>(revAlign.score1);
    } else {
        score = rawSmithWatermanScore;
    }
    align.evalue = evaluer.computeEvalueCorr(score, queryMuLambda.first, queryMuLambda.second, querySeqLen, targetSeqLen, targetMuLambda);
    hasLowerEvalue = align.evalue > par.evalThr;
    if (hasLowerEvalue) {
        return -1;
    }

    bool blockAlignFailed = false;
    if (structureSmithWaterman.isProfileSearch() == false) {
        StructureSmithWaterman::s_align alignTmp = structureSmithWaterman.alignStartPosBacktraceBlock(
            tSeqAA.numSequence, tSeq3Di.numSequence, targetSeqLen, par.gapOpen.values.aminoacid(),
            par.gapExtend.values.aminoacid(), backtrace, align,
            (tSeq12St != NULL) ? tSeq12St->numSequence : NULL
        );

        if (alignTmp.score1 == UINT32_MAX) {
            Debug(Debug::WARNING) << "block-align failed, falling back to normal alignment\n";
            blockAlignFailed = true;
        } else {
            align = alignTmp;
        }
    }

    if (blockAlignFailed || structureSmithWaterman.isProfileSearch()) {
        align = structureSmithWaterman.alignStartPosBacktrace<StructureSmithWaterman::PROFILE>(tSeqAA.numSequence,
                                                                                               tSeq3Di.numSequence,
                                                                                               targetSeqLen,
                                                                                               par.gapOpen.values.aminoacid(),
                                                                                               par.gapExtend.values.aminoacid(),
                                                                                               par.alignmentMode,
                                                                                               backtrace, align,
                                                                                               par.covMode, par.covThr,
                                                                                               querySeqLen / 2,
                                                                                               db12StSeq);
    }

    unsigned int alnLength = Matcher::computeAlnLength(align.qStartPos1, align.qEndPos1, align.dbStartPos1, align.dbEndPos1);
    if(backtrace.size() > 0){
        alnLength = backtrace.size();
        seqId = Util::computeSeqId(par.seqIdMode, align.identicalAACnt, querySeqLen, targetSeqLen, alnLength);
    }
    align.score1 = score;
    res = Matcher::result_t(tSeqAA.getDbKey(), align.score1, align.qCov, align.tCov, seqId, align.evalue, alnLength,
                            align.qStartPos1, align.qEndPos1, querySeqLen, align.dbStartPos1, align.dbEndPos1, targetSeqLen, backtrace);
    return 0;
}


int computeAlternativeAlignment(StructureSmithWaterman & structureSmithWaterman,
                                StructureSmithWaterman & reverseStructureSmithWaterman,
                                Sequence & tSeqAA, Sequence & tSeq3Di,
                                Sequence * tSeq12St,
                                unsigned int querySeqLen, unsigned int targetSeqLen,
                                EvalueNeuralNet & evaluer, std::pair<double, double> queryMuLambda,
                                const std::pair<double, double> *targetMuLambda,
                                Matcher::result_t & result, Matcher::result_t & altRes,
                                std::string & backtrace, Parameters & par, bool useReverseScore) {
    const unsigned char xAAIndex = tSeqAA.subMat->aa2num[static_cast<int>('X')];
    const unsigned char x3DiIndex = tSeq3Di.subMat->aa2num[static_cast<int>('X')];
    for (int pos = result.dbStartPos; pos < result.dbEndPos; ++pos) {
        tSeqAA.numSequence[pos] = xAAIndex;
        tSeq3Di.numSequence[pos] = x3DiIndex;
        if (tSeq12St != NULL) {
            tSeq12St->numSequence[pos] = tSeq12St->subMat->aa2num[static_cast<int>('X')];
        }
    }
    if (alignStructure(structureSmithWaterman, reverseStructureSmithWaterman,
                       tSeqAA, tSeq3Di, tSeq12St, querySeqLen, targetSeqLen,
                       evaluer, queryMuLambda, targetMuLambda, altRes, backtrace, par, useReverseScore) == -1) {
        return -1;
    }
    if (Alignment::checkCriteria(altRes, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr)) {
        return 0;
    } else {
        return -1;
    }
}


int structurealign(int argc, const char **argv, const Command& command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    structureAlignDefault(par);
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_ALIGN);

    const bool touch = (par.preloadMode != Parameters::PRELOAD_MODE_MMAP);

    bool sameDB = false;
    uint16_t extended = DBReader<unsigned int>::getExtendedDbtype(FileUtil::parseDbType(par.db3.c_str()));
    bool alignmentIsExtended = extended & Parameters::DBTYPE_EXTENDED_INDEX_NEED_SRC;
    IndexReader tAADbr(par.db2, par.threads,
                             alignmentIsExtended ? IndexReader::SRC_SEQUENCES : IndexReader::SEQUENCES,
                             (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);

    std::string t3DiDbrName =  StructureUtil::getIndexWithSuffix(par.db2, "_ss");
    bool is3DiIdx = Parameters::isEqualDbtype(FileUtil::parseDbType(t3DiDbrName.c_str()),
                                              Parameters::DBTYPE_INDEX_DB);

    IndexReader t3DiDbr(is3DiIdx ? t3DiDbrName : par.db2, par.threads,
                              alignmentIsExtended ? IndexReader::SRC_SEQUENCES : IndexReader::SEQUENCES,
                              (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0,
                              DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA,
                              alignmentIsExtended ? "_seq_ss" : "_ss");

    IndexReader *q3DiDbr = NULL;
    IndexReader *qAADbr = NULL;
    IndexReader *q12StDbr = NULL;

    if (par.db1.compare(par.db2) == 0) {
        sameDB = true;
        q3DiDbr = &t3DiDbr;
        qAADbr = &tAADbr;
    } else {
        qAADbr = new IndexReader(par.db1, par.threads, IndexReader::SRC_SEQUENCES, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);
        q3DiDbr = new IndexReader(StructureUtil::getIndexWithSuffix(par.db1, "_ss"), par.threads, IndexReader::SRC_SEQUENCES, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);
    }

    bool target3Di12St = StructureUtil::is3Di12StDb(t3DiDbr.sequenceReader->getDbtype());
    bool query3Di12St  = StructureUtil::is3Di12StDb(q3DiDbr->sequenceReader->getDbtype());
    if (!query3Di12St && FileUtil::fileExists((par.db1 + "_ss12.dbtype").c_str())) {
        q12StDbr = new IndexReader(StructureUtil::getIndexWithSuffix(par.db1, "_ss12"), par.threads,
                                   IndexReader::SRC_SEQUENCES,
                                   (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);
    }
    bool queryHas12St = query3Di12St || (q12StDbr != NULL);
    bool use12StScoring = par.useAuxScoring && queryHas12St && target3Di12St;
    const int evalueNNMode = (par.evalueNNMode == LocalParameters::EVALUE_NN_MODE_AUTO)
        ? (use12StScoring ? LocalParameters::EVALUE_NN_MODE_3DI_12ST : LocalParameters::EVALUE_NN_MODE_3DI)
        : par.evalueNNMode;
    bool use12StEvalue = evalueNNMode == LocalParameters::EVALUE_NN_MODE_3DI_12ST;
    if (use12StEvalue && !queryHas12St) {
        Debug(Debug::ERROR) << "--evalue-nn-mode 2 requires a query 12-state alphabet in the packed _ss DB or a _ss12 DB\n";
        EXIT(EXIT_FAILURE);
    }
    if (use12StEvalue && !target3Di12St) {
        Debug(Debug::ERROR) << "--evalue-nn-mode 2 requires a target 12-state alphabet in the packed _ss DB\n";
        EXIT(EXIT_FAILURE);
    }
    bool db1CaExist = FileUtil::fileExists((par.db1 + "_ca.dbtype").c_str());
    bool db2CaExist = FileUtil::fileExists((par.db2 + "_ca.dbtype").c_str());
    if(Parameters::isEqualDbtype(tAADbr.getDbtype(), Parameters::DBTYPE_INDEX_DB)){
        db2CaExist = true;
    }
    bool useRankModel = par.sortByStructureBits == 2;
    if (useRankModel && (par.alignmentMode == 1 || par.alignmentMode == 2)) {
        Debug(Debug::WARNING) << "--sort-by-structure-bits 2 needs an alignment mode with backtraces\n"
                              << "Hits are sorted by E-value\n";
        useRankModel = false;
    }
    HitRanker::Model rankModel;
    if (useRankModel) {
        rankModel.loadBuiltin(db1CaExist, db2CaExist);
        // the ranker sorts hits itself and keeps the raw bit score
        par.sortByStructureBits = 0;
    } else if (par.sortByStructureBits == 2) {
        par.sortByStructureBits = 0;
    }

    if(par.sortByStructureBits) {
        bool disableStructureBits = false;
        if(db1CaExist == false || db2CaExist == false){
            Debug(Debug::WARNING) << "Cannot find " << FileUtil::baseName(par.db1) << " C-alpha or " << FileUtil::baseName(par.db2) << " C-alpha database\n";
            disableStructureBits = true;
        }
        if(par.alignmentMode == 1 || par.alignmentMode == 2){
            Debug(Debug::WARNING) << "Cannot use --sort-by-structure-bits 1 with --alignment-mode 1 or 2\n";
            disableStructureBits = true;
        }
        if(disableStructureBits){
            Debug(Debug::WARNING) << "Disabling --sort-by-structure-bits\n";
            Debug(Debug::WARNING) << "This impacts the final score and ranking of hits, but not E-values themselves. Ranking alterations primarily occur for E-values < 10^-1.\n";
            par.sortByStructureBits = false;
        }
    }

    DBReader<unsigned int> resultReader(par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<unsigned int>::USE_DATA|DBReader<unsigned int>::USE_INDEX);
    resultReader.open(DBReader<unsigned int>::LINEAR_ACCCESS);

    int dbtype =  Parameters::DBTYPE_ALIGNMENT_RES;
    if(alignmentIsExtended){
        dbtype = DBReader<unsigned int>::setExtendedDbtype(dbtype, Parameters::DBTYPE_EXTENDED_INDEX_NEED_SRC);
    }
    DBWriter dbw(par.db4.c_str(), par.db4Index.c_str(), static_cast<unsigned int>(par.threads), par.compressed,  dbtype);
    dbw.open();

    bool needTMaligner = (par.tmScoreThr > 0);
    bool needLDDT = (par.lddtThr > 0);
    if (par.sortByStructureBits) {
        needLDDT = true;
        needTMaligner = true;
    } else {
        if (needTMaligner && (db1CaExist == false || db2CaExist == false)) {
            Debug(Debug::WARNING) << "Cannot use --tmscore-threshold with --sort-by-structure-bits 0\n"
                                  << "Disabling --tmscore-threshold\n";
            needTMaligner = false;
        }
        if (needLDDT && (db1CaExist == false || db2CaExist == false)) {
            Debug(Debug::WARNING) << "Cannot use --lddt-threshold with --sort-by-structure-bits 0\n"
                                  << "Disabling --lddt-threshold\n";
            needLDDT = false;
        }
    }
    if (useRankModel && db1CaExist && db2CaExist) {
        // native lDDT and TM-score are ranking features when query coordinates exist
        needTMaligner = true;
        needLDDT = true;
    }
    bool needCalpha = (needTMaligner || needLDDT);
    bool needTargetCalpha = needCalpha || (useRankModel && db2CaExist);
    IndexReader *qcadbr = NULL;
    IndexReader *tcadbr = NULL;
    if(needCalpha || (sameDB && needTargetCalpha)){
        qcadbr = new IndexReader(
                par.db1,
                par.threads,
                IndexReader::makeUserDatabaseType(LocalParameters::INDEX_DB_CA_KEY_DB1),
                touch ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0,
                DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA,
                "_ca");
    }
    if(needTargetCalpha){
        if (sameDB) {
            tcadbr = qcadbr;
        } else {
             tcadbr = new IndexReader(
                    par.db2,
                    par.threads,
                    alignmentIsExtended ? IndexReader::makeUserDatabaseType(LocalParameters::INDEX_DB_CA_KEY_DB2) :
                                           IndexReader::makeUserDatabaseType(LocalParameters::INDEX_DB_CA_KEY_DB1),
                    touch ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0,
                    DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA,
                    alignmentIsExtended ? "_seq_ca" : "_ca"
            );
        }
    }

    SubstitutionMatrix subMat3Di(par.scoringMatrixFile.values.aminoacid().c_str(), 2.1, par.scoreBias);
    std::string blosum;
    for (size_t i = 0; i < par.substitutionMatrices.size(); i++) {
        if (par.substitutionMatrices[i].name == "blosum62.out") {
            std::string matrixData((const char *)par.substitutionMatrices[i].subMatData, par.substitutionMatrices[i].subMatDataLen);
            std::string matrixName = par.substitutionMatrices[i].name;
            char * serializedMatrix = BaseMatrix::serialize(matrixName, matrixData);
            blosum.assign(serializedMatrix);
            free(serializedMatrix);
            break;
        }
    }
    std::string mat12st;
    if (queryHas12St || target3Di12St) {
        for (size_t i = 0; i < par.substitutionMatrices.size(); i++) {
            if (par.substitutionMatrices[i].name == "12st.out") {
                std::string matrixData((const char *)par.substitutionMatrices[i].subMatData,
                                       par.substitutionMatrices[i].subMatDataLen);
                std::string matrixName = par.substitutionMatrices[i].name;
                char * serializedMatrix = BaseMatrix::serialize(matrixName, matrixData);
                mat12st.assign(serializedMatrix);
                free(serializedMatrix);
                break;
            }
        }
        if (mat12st.empty()) {
            Debug(Debug::ERROR) << "Cannot find 12st substitution matrix\n";
            EXIT(EXIT_FAILURE);
        }
    }
    float aaFactor = (par.alignmentType == LocalParameters::ALIGNMENT_TYPE_3DI_AA) ? 1.4 : 0.0;
    SubstitutionMatrix subMatAA(blosum.c_str(), aaFactor, par.scoreBias);
    SubstitutionMatrix *subMat12St = NULL;
    if (queryHas12St || target3Di12St) {
        subMat12St = new SubstitutionMatrix(mat12st.c_str(), par.submat12stScale, par.scoreBias);
    }
    //temporary output file
    Debug::Progress progress(resultReader.getSize());

    // sub. mat needed for query profile
    int8_t * tinySubMatAA = (int8_t*) mem_align(ALIGN_INT, subMatAA.alphabetSize * 32);
    int8_t * tinySubMat3Di = (int8_t*) mem_align(ALIGN_INT, subMat3Di.alphabetSize * 32);

    for (int i = 0; i < subMat3Di.alphabetSize; i++) {
        for (int j = 0; j < subMat3Di.alphabetSize; j++) {
            tinySubMat3Di[i * subMat3Di.alphabetSize + j] = subMat3Di.subMatrix[i][j]; // for farrar profile
        }
    }
    for (int i = 0; i < subMatAA.alphabetSize; i++) {
        for (int j = 0; j < subMatAA.alphabetSize; j++) {
            tinySubMatAA[i * subMatAA.alphabetSize + j] = subMatAA.subMatrix[i][j];
        }
    }

#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        EvalueNeuralNet evaluer(tAADbr.sequenceReader->getAminoAcidDBSize(),
                                tAADbr.sequenceReader->getSize(),
                                &subMat3Di,
                                evalueNNMode,
                                par.evalue12StProfileComp != 0,
                                subMat12St);
        std::vector<Matcher::result_t> alignmentResult;
        StructureSmithWaterman structureSmithWaterman(par.maxSeqLen, subMat3Di.alphabetSize, par.compBiasCorrection, par.compBiasCorrectionScale, &subMatAA, &subMat3Di);
        StructureSmithWaterman reverseStructureSmithWaterman(par.maxSeqLen, subMat3Di.alphabetSize, par.compBiasCorrection, par.compBiasCorrectionScale, &subMatAA, &subMat3Di);
        TMaligner *tmaligner = NULL;
        if(needTMaligner) {
            tmaligner = new TMaligner(
                    std::max(q3DiDbr->sequenceReader->getMaxSeqLen() + 1, t3DiDbr.sequenceReader->getMaxSeqLen() + 1), false, true, par.exactTMscore);
        }
        LDDTCalculator *lddtcalculator = NULL;
        if(needLDDT) {
            lddtcalculator = new LDDTCalculator(q3DiDbr->sequenceReader->getMaxSeqLen() + 1,  t3DiDbr.sequenceReader->getMaxSeqLen() + 1);
        }
        Sequence qSeqAA(par.maxSeqLen, qAADbr->getDbtype(), (const BaseMatrix *) &subMatAA, 0, false, par.compBiasCorrection);
        Sequence qSeq3Di(par.maxSeqLen, query3Di12St ? Parameters::DBTYPE_AMINO_ACIDS : q3DiDbr->getDbtype(), (const BaseMatrix *) &subMat3Di, 0, false, par.compBiasCorrection);
        Sequence tSeqAA(par.maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, (const BaseMatrix *) &subMatAA, 0, false, par.compBiasCorrection);
        Sequence tSeq3Di(par.maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, (const BaseMatrix *) &subMat3Di, 0, false, par.compBiasCorrection);
        std::unique_ptr<Sequence> qSeq12St;
        std::unique_ptr<Sequence> qSeq12StUnmasked;
        std::unique_ptr<Sequence> tSeq12St;
        const bool query12StProfileLayout = (use12StScoring || use12StEvalue)
            && q12StDbr != NULL
            && Parameters::isEqualDbtype(q12StDbr->getDbtype(), Parameters::DBTYPE_HMM_PROFILE);
        int8_t *tinySubMat12St = NULL;
        if (subMat12St) {
            int stride = subMatAA.alphabetSize;
            tinySubMat12St = (int8_t*) mem_align(ALIGN_INT, stride * stride * sizeof(int8_t));
            memset(tinySubMat12St, 0, stride * stride * sizeof(int8_t));
            for (int i = 0; i < subMat12St->alphabetSize; i++) {
                const unsigned char rowLetter = static_cast<unsigned char>(subMat12St->num2aa[i]);
                const int rowIndex = query12StProfileLayout ? subMatAA.aa2num[rowLetter] : i;
                for (int j = 0; j < subMat12St->alphabetSize; j++) {
                    const unsigned char colLetter = static_cast<unsigned char>(subMat12St->num2aa[j]);
                    const int colIndex = query12StProfileLayout ? subMatAA.aa2num[colLetter] : j;
                    tinySubMat12St[rowIndex * stride + colIndex] = subMat12St->subMatrix[i][j];
                }
            }
        }
        std::vector<char> qSeq3Di21Buf;
        std::vector<char> qSeq12StBuf;
        std::vector<char> qSeq12StUnmaskedBuf;
        std::vector<char> tSeq3Di21Buf;
        std::vector<char> tSeq12StBuf;
        if (queryHas12St) {
            qSeq3Di21Buf.reserve(par.maxSeqLen);
            qSeq12StBuf.reserve(par.maxSeqLen);
            if (use12StScoring || use12StEvalue) {
                int q12StDbtype = query3Di12St ? Parameters::DBTYPE_AMINO_ACIDS : q12StDbr->getDbtype();
                const BaseMatrix *q12StMat = (q12StDbtype == Parameters::DBTYPE_HMM_PROFILE)
                    ? static_cast<const BaseMatrix *>(&subMatAA)
                    : static_cast<const BaseMatrix *>(subMat12St);
                qSeq12St.reset(new Sequence(par.maxSeqLen, q12StDbtype, q12StMat, 0, false, par.compBiasCorrection));
            }
            if (use12StEvalue && query3Di12St) {
                qSeq12StUnmaskedBuf.reserve(par.maxSeqLen);
                qSeq12StUnmasked.reset(new Sequence(par.maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, subMat12St, 0, false, par.compBiasCorrection));
            }
        }
        if (target3Di12St) {
            tSeq3Di21Buf.reserve(par.maxSeqLen);
            tSeq12StBuf.reserve(par.maxSeqLen);
            if (use12StScoring || use12StEvalue) {
                const BaseMatrix *t12StMat = query12StProfileLayout
                    ? static_cast<const BaseMatrix *>(&subMatAA)
                    : static_cast<const BaseMatrix *>(subMat12St);
                tSeq12St.reset(new Sequence(par.maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, t12StMat, 0, false, par.compBiasCorrection));
            }
        }
        std::string backtrace;
        char buffer[1024+32768];
        std::string resultBuffer;

        Coordinate16 qcoords;
        Coordinate16 tcoords;

        TMaligner::TMscoreResult tmres;
        LDDTCalculator::LDDTScoreResult lddtres;

        HitRanker::Chain queryChain;
        HitRanker::Chain targetChain;
        std::vector<unsigned char> queryAA;
        std::vector<unsigned char> query3Di;
        std::vector<float> rankFeatures(HitRanker::FEATURE_NAMES.size());
        std::vector<float> rankFeatureRows;     // one row per scored hit
        std::vector<float> rankFeatureColumns;
        std::vector<size_t> rankRowOfResult;    // row of each entry in alignmentResult
        std::vector<double> rankScores;
        // write output file

#pragma omp for schedule(dynamic, 1)
        for (size_t id = 0; id < resultReader.getSize(); id++) {
            progress.updateProgress();
            char *data = resultReader.getData(id, thread_idx);
            size_t queryKey = resultReader.getDbKey(id);
            if(*data != '\0') {
                unsigned int queryId = q3DiDbr->sequenceReader->getId(queryKey);

                char *querySeqAA = qAADbr->sequenceReader->getData(queryId, thread_idx);
                char *querySeq3Di = q3DiDbr->sequenceReader->getData(queryId, thread_idx);
                unsigned int querySeqLen = q3DiDbr->sequenceReader->getSeqLen(queryId);
                const char *querySeq3Di21 = querySeq3Di;
                Sequence *qSeq12StForEvalue = qSeq12St.get();
                if (query3Di12St) {
                    if (qSeq12StUnmasked) {
                        StructureUtil::split3Di12St(querySeq3Di, querySeqLen, qSeq3Di21Buf, qSeq12StUnmaskedBuf, subMat3Di, *subMat12St, false);
                        qSeq12StUnmasked->mapSequence(id, queryKey, qSeq12StUnmaskedBuf.data(), querySeqLen);
                        qSeq12StForEvalue = qSeq12StUnmasked.get();
                        qSeq12StBuf.assign(qSeq12StUnmaskedBuf.begin(), qSeq12StUnmaskedBuf.begin() + querySeqLen);
                        StructureUtil::mask12StSeqs(qSeq3Di21Buf.data(), qSeq12StBuf.data(), querySeqLen, true);
                    } else {
                        StructureUtil::split3Di12St(querySeq3Di, querySeqLen, qSeq3Di21Buf, qSeq12StBuf, subMat3Di, *subMat12St, true);
                    }
                    querySeq3Di21 = qSeq3Di21Buf.data();
                    if (use12StScoring || use12StEvalue) {
                        qSeq12St->mapSequence(id, queryKey, qSeq12StBuf.data(), querySeqLen);
                    }
                } else if (use12StScoring || use12StEvalue) {
                    unsigned int query12StId = q12StDbr->sequenceReader->getId(queryKey);
                    if (query12StId == UINT_MAX) {
                        Debug(Debug::ERROR) << "Query " << queryKey << " is missing in the _ss12 database\n";
                        EXIT(EXIT_FAILURE);
                    }
                    char *querySeq12St = q12StDbr->sequenceReader->getData(query12StId, thread_idx);
                    qSeq12St->mapSequence(id, queryKey, querySeq12St, querySeqLen);
                    if (qSeq12St->L != static_cast<int>(querySeqLen)) {
                        Debug(Debug::ERROR) << "Query " << queryKey << " has a different length in the _ss12 database\n";
                        EXIT(EXIT_FAILURE);
                    }
                }
                qSeq3Di.mapSequence(id, queryKey, querySeq3Di21, querySeqLen);
                qSeqAA.mapSequence(id, queryKey, querySeqAA, querySeqLen);
                if (useRankModel) {
                    // copy, the query sequences get reversed below
                    queryAA.assign(qSeqAA.numSequence, qSeqAA.numSequence + qSeqAA.L);
                    query3Di.assign(qSeq3Di.numSequence, qSeq3Di.numSequence + qSeq3Di.L);
                    queryChain.seqAA = queryAA.data();
                    queryChain.seq3Di = query3Di.data();
                    queryChain.length = qSeq3Di.L;
                    queryChain.x = queryChain.y = queryChain.z = NULL;
                    queryChain.setSecondaryStructure(subMat3Di);
                }
                if (!query3Di12St && use12StScoring && query12StProfileLayout
                    && Parameters::isEqualDbtype(q3DiDbr->getDbtype(), Parameters::DBTYPE_HMM_PROFILE)) {
                    StructureUtil::mask12StProfile(qSeq3Di, *qSeq12St);
                }
                if(needCalpha){
                    size_t qId = qcadbr->sequenceReader->getId(queryKey);
                    char *qcadata = qcadbr->sequenceReader->getData(qId, thread_idx);
                    size_t qCaLength = qcadbr->sequenceReader->getEntryLen(qId);
                    float* queryCaData = qcoords.read(qcadata, qSeq3Di.L, qCaLength);
                    if(needTMaligner){
                        tmaligner->initQuery(queryCaData, &queryCaData[qSeq3Di.L], &queryCaData[qSeq3Di.L+qSeq3Di.L], NULL, qSeq3Di.L);
                    }
                    if(needLDDT){
                        lddtcalculator->initQuery(qSeq3Di.L, queryCaData, &queryCaData[qSeq3Di.L], &queryCaData[qSeq3Di.L+qSeq3Di.L]);
                    }
                }
                std::pair<double, double> muLambda = evaluer.uses12StModel()
                    ? evaluer.predictMuLambda12St(qSeq3Di, *qSeq12StForEvalue)
                    : evaluer.predictMuLambda(qSeq3Di.numSequence, qSeq3Di.L);
                structureSmithWaterman.ssw_init(&qSeqAA, &qSeq3Di, tinySubMatAA, tinySubMat3Di, &subMatAA,
                                               use12StScoring ? qSeq12St.get() : NULL, use12StScoring ? tinySubMat12St : NULL,
                                               use12StScoring ? static_cast<const BaseMatrix *>(subMat12St) : NULL);
                if (par.useReverseScore) {
                    qSeq3Di.reverse();
                    qSeqAA.reverse();
                    if (use12StScoring) {
                        qSeq12St->reverse();
                    }
                    reverseStructureSmithWaterman.ssw_init(&qSeqAA, &qSeq3Di, tinySubMatAA, tinySubMat3Di, &subMatAA,
                                                           use12StScoring ? qSeq12St.get() : NULL, use12StScoring ? tinySubMat12St : NULL,
                                                           use12StScoring ? static_cast<const BaseMatrix *>(subMat12St) : NULL);
                    qSeq3Di.reverse();
                    qSeqAA.reverse();
                    if (use12StScoring) {
                        qSeq12St->reverse();
                    }
                }
                int passedNum = 0;
                int rejected = 0;
                while (*data != '\0' && passedNum < par.maxAccept && rejected < par.maxRejected) {
                    char dbKeyBuffer[255 + 1];
                    Util::parseKey(data, dbKeyBuffer);
                    data = Util::skipLine(data);
                    const unsigned int dbKey = (unsigned int) strtoul(dbKeyBuffer, NULL, 10);
                    unsigned int targetId = t3DiDbr.sequenceReader->getId(dbKey);
                    const bool isIdentity = (queryId == targetId && (par.includeIdentity || sameDB))? true : false;

                    char * targetSeq3Di = t3DiDbr.sequenceReader->getData(targetId, thread_idx);
                    char * targetSeqAA = tAADbr.sequenceReader->getData(targetId, thread_idx);
                    const int targetSeqLen = static_cast<int>(t3DiDbr.sequenceReader->getSeqLen(targetId));

                    const char *targetSeq3Di21 = targetSeq3Di;
                    if (target3Di12St) {
                        StructureUtil::split3Di12St(targetSeq3Di, targetSeqLen, tSeq3Di21Buf, tSeq12StBuf, subMat3Di, *subMat12St);
                        targetSeq3Di21 = tSeq3Di21Buf.data();
                        if (use12StScoring || use12StEvalue) {
                            tSeq12St->mapSequence(targetId, dbKey, tSeq12StBuf.data(), targetSeqLen);
                        }
                    }
                    tSeq3Di.mapSequence(targetId, dbKey, targetSeq3Di21, targetSeqLen);
                    tSeqAA.mapSequence(targetId, dbKey, targetSeqAA, targetSeqLen);
                    if(Util::canBeCovered(par.covThr, par.covMode, qSeq3Di.L, targetSeqLen) == false){
                        rejected++;
                        continue;
                    }
                    std::pair<double, double> targetMuLambda;
                    const std::pair<double, double> *targetMuLambdaPtr = NULL;
                    if (evaluer.uses12StModel() && tSeq12St) {
                        targetMuLambda = evaluer.predictMuLambda12St(tSeq3Di, *tSeq12St);
                        targetMuLambdaPtr = &targetMuLambda;
                    }
                    Matcher::result_t res;
                    if(alignStructure(structureSmithWaterman, reverseStructureSmithWaterman,
                                      tSeqAA, tSeq3Di, use12StScoring ? tSeq12St.get() : NULL,
                                      querySeqLen, targetSeqLen,
                                      evaluer, muLambda, targetMuLambdaPtr, res, backtrace, par, par.useReverseScore) == -1){
                        rejected++;
                        continue;
                    }

                    if (Alignment::checkCriteria(res, isIdentity, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr)) {
                        float* targetCaData = NULL;
                        if(needTargetCalpha) {
                            size_t tId = tcadbr->sequenceReader->getId(res.dbKey);
                            char *tcadata = tcadbr->sequenceReader->getData(tId, thread_idx);
                            size_t tCaLength = tcadbr->sequenceReader->getEntryLen(tId);
                            targetCaData = tcoords.read(tcadata, res.dbLen, tCaLength);
                        }
                        if(needCalpha) {
                            if(needTMaligner) {
                                tmres = tmaligner->computeTMscore(targetCaData,
                                                                  &targetCaData[res.dbLen],
                                                                  &targetCaData[res.dbLen +
                                                                                res.dbLen],
                                                                  res.dbLen,
                                                                  res.qStartPos,
                                                                  res.dbStartPos,
                                                                  res.backtrace,
                                                                  TMaligner::normalization(par.tmScoreThrMode, std::min(res.qEndPos - res.qStartPos, res.dbEndPos - res.dbStartPos ), res.qLen, res.dbLen));

                                if (tmres.tmscore < par.tmScoreThr) {
                                    continue;
                                }
                            }
                            if(needLDDT){
                                lddtres = lddtcalculator->computeLDDTScore(res.dbLen, res.qStartPos, res.dbStartPos,
                                                                           res.backtrace,
                                                                           targetCaData, &targetCaData[res.dbLen],
                                                                           &targetCaData[res.dbLen+res.dbLen]);

                                if(lddtres.avgLddtScore < par.lddtThr){
                                    continue;
                                }
                                res.dbcov = lddtres.avgLddtScore;
                            }
                            if(par.sortByStructureBits && needTMaligner && needLDDT){
                                res.score = res.score * sqrt(lddtres.avgLddtScore * tmres.tmscore);
                            }
                        }
                        const size_t rankRow = rankFeatureRows.size() / HitRanker::FEATURE_NAMES.size();
                        if (useRankModel) {
                            targetChain.seqAA = tSeqAA.numSequence;
                            targetChain.seq3Di = tSeq3Di.numSequence;
                            targetChain.length = targetSeqLen;
                            targetChain.x = targetCaData;
                            targetChain.y = targetCaData != NULL ? &targetCaData[res.dbLen] : NULL;
                            targetChain.z = targetCaData != NULL ? &targetCaData[res.dbLen + res.dbLen] : NULL;
                            targetChain.setSecondaryStructure(subMat3Di);
                            const bool hasNative = needTMaligner && needLDDT;
                            HitRanker::computeFeatures(queryChain, targetChain, res, subMatAA, subMat3Di,
                                                       hasNative ? lddtres.avgLddtScore : NAN,
                                                       hasNative ? tmres.tmscore : NAN, rankFeatures.data());
                            rankFeatureRows.insert(rankFeatureRows.end(), rankFeatures.begin(), rankFeatures.end());
                        }

                        alignmentResult.emplace_back(res);
                        rankRowOfResult.push_back(rankRow);
                        int altAli = par.altAlignment;
                        bool moreAltAli = true;
                        while(altAli && moreAltAli){
                            Matcher::result_t altRes;
                            if(computeAlternativeAlignment(structureSmithWaterman, reverseStructureSmithWaterman,
                                                           tSeqAA, tSeq3Di, use12StScoring ? tSeq12St.get() : NULL,
                                                           querySeqLen, targetSeqLen,
                                                           evaluer, muLambda, targetMuLambdaPtr, res, altRes,
                                                           backtrace, par, par.useReverseScore) == -1) {
                                moreAltAli = false;
                                continue;
                            }
                            alignmentResult.push_back(altRes);
                            // alternative alignments keep the score of their primary hit
                            rankRowOfResult.push_back(rankRow);
                            res = altRes;
                            altAli--;
                        }
                        passedNum++;
                        rejected = 0;
                    } else {
                        rejected++;
                    }
                }
            }


            if (useRankModel && alignmentResult.size() > 1) {
                const size_t nFeatures = HitRanker::FEATURE_NAMES.size();
                const size_t nRows = rankFeatureRows.size() / nFeatures;
                rankFeatureColumns.resize(rankFeatureRows.size());
                for (size_t r = 0; r < nRows; r++) {
                    for (size_t f = 0; f < nFeatures; f++) {
                        rankFeatureColumns[f * nRows + r] = rankFeatureRows[r * nFeatures + f];
                    }
                }
                std::vector<double> rowScores(nRows);
                rankModel.predict(rankFeatureColumns.data(), nRows, rowScores.data());
                rankScores.resize(alignmentResult.size());
                for (size_t i = 0; i < alignmentResult.size(); i++) {
                    rankScores[i] = rowScores[rankRowOfResult[i]];
                }
                std::vector<size_t> order(alignmentResult.size());
                for (size_t i = 0; i < order.size(); i++) {
                    order[i] = i;
                }
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    if (rankScores[a] != rankScores[b]) {
                        return rankScores[a] > rankScores[b];
                    }
                    return Matcher::compareHits(alignmentResult[a], alignmentResult[b]);
                });
                std::vector<Matcher::result_t> sorted;
                sorted.reserve(order.size());
                for (size_t i : order) {
                    sorted.emplace_back(alignmentResult[i]);
                }
                alignmentResult.swap(sorted);
            } else if (alignmentResult.size() > 1) {
                if(par.sortByStructureBits) {
                    SORT_SERIAL(alignmentResult.begin(), alignmentResult.end(), compareHitsByStructureBits);
                } else {
                    SORT_SERIAL(alignmentResult.begin(), alignmentResult.end(), Matcher::compareHits);
                }
            }
            for (size_t result = 0; result < alignmentResult.size(); result++) {
                size_t len = Matcher::resultToBuffer(buffer, alignmentResult[result], par.addBacktrace);
                resultBuffer.append(buffer, len);
            }
            dbw.writeData(resultBuffer.c_str(), resultBuffer.length(), queryKey, thread_idx);
            resultBuffer.clear();
            alignmentResult.clear();
            rankScores.clear();
            rankRowOfResult.clear();
            rankFeatureRows.clear();
        }
        if(needTMaligner){
            delete tmaligner;
        }
        if(needLDDT){
            delete lddtcalculator;
        }
        free(tinySubMat12St);
    }

    free(tinySubMatAA);
    free(tinySubMat3Di);
    delete subMat12St;

    dbw.close();
    resultReader.close();

    if (sameDB == false && tcadbr != NULL) {
        delete tcadbr;
    }
    if (qcadbr != NULL) {
        delete qcadbr;
    }

    if (sameDB == false) {
        delete q3DiDbr;
        delete qAADbr;
    }
    if (q12StDbr) {
        delete q12StDbr;
    }

    return EXIT_SUCCESS;
}
