#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "LocalParameters.h"
#include "Coordinate16.h"
#include "PulchraWrapper.h"
#include "structureto3di.h"
#include "structureto12st.h"
#include "SubstitutionMatrix.h"
#include "FileUtil.h"

#ifdef OPENMP
#include <omp.h>
#endif

int add12st(int argc, const char **argv, const Command &command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_COMMON);

    std::string dbBase = par.db1;
    std::string dbSs = dbBase + "_ss";
    std::string dbSsIndex = dbBase + "_ss.index";
    std::string dbSsDbtype = dbBase + "_ss.dbtype";
    std::string dbCa = dbBase + "_ca";
    std::string dbCaIndex = dbBase + "_ca.index";

    // Check that _ss exists and is old format (no 12st)
    DBReader<unsigned int> ssReader(dbSs.c_str(), dbSsIndex.c_str(), par.threads,
                                    DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    ssReader.open(DBReader<unsigned int>::NOSORT);

    if ((DBReader<unsigned int>::getExtendedDbtype(ssReader.getDbtype()) & LocalParameters::DBTYPE_EXTENDED_3DI_12ST) != 0) {
        Debug(Debug::INFO) << "Database already contains 12-state alphabet. Nothing to do.\n";
        ssReader.close();
        return EXIT_SUCCESS;
    }

    // Open AA sequence db (needed for PULCHRA amino acid input)
    DBReader<unsigned int> aaReader(dbBase.c_str(), (dbBase + ".index").c_str(), par.threads,
                                    DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    aaReader.open(DBReader<unsigned int>::NOSORT);

    // Open CA coordinate db
    DBReader<unsigned int> caReader(dbCa.c_str(), dbCaIndex.c_str(), par.threads,
                                    DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    caReader.open(DBReader<unsigned int>::NOSORT);

    // Move old _ss to _ss_old
    std::string dbSsOld = dbBase + "_ss_old";
    std::string dbSsOldIndex = dbBase + "_ss_old.index";
    std::string dbSsOldDbtype = dbBase + "_ss_old.dbtype";
    FileUtil::move(dbSs.c_str(), dbSsOld.c_str());
    FileUtil::move(dbSsIndex.c_str(), dbSsOldIndex.c_str());
    FileUtil::move(dbSsDbtype.c_str(), dbSsOldDbtype.c_str());

    // Re-open old _ss from its new location
    ssReader.close();
    DBReader<unsigned int> ssOldReader(dbSsOld.c_str(), dbSsOldIndex.c_str(), par.threads,
                                       DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    ssOldReader.open(DBReader<unsigned int>::NOSORT);

    int ssDbtype = DBReader<unsigned int>::setExtendedDbtype(Parameters::DBTYPE_AMINO_ACIDS,
                                                             LocalParameters::DBTYPE_EXTENDED_3DI_12ST);
    DBWriter ssWriter(dbSs.c_str(), dbSsIndex.c_str(), par.threads, par.compressed, ssDbtype);
    ssWriter.open();

    SubstitutionMatrix mat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);

    Debug::Progress progress(ssOldReader.getSize());

#pragma omp parallel num_threads(par.threads)
    {
        int thread_idx = 0;
#ifdef OPENMP
        thread_idx = omp_get_thread_num();
#endif
        PulchraWrapper pulchra;
        StructureTo12St structureTo12St;
        Coordinate16 coords;

        std::vector<Vec3> caVec;
        std::vector<Vec3> nVec;
        std::vector<Vec3> cVec;
        std::vector<Vec3> cbVec;
        std::vector<char> amiVec;
        std::string result;

#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < ssOldReader.getSize(); i++) {
            progress.updateProgress();

            unsigned int key = ssOldReader.getDbKey(i);

            // Read old 3Di sequence
            const char *ssData = ssOldReader.getData(i, thread_idx);
            size_t seqLen = std::max(ssOldReader.getEntryLen(i), (size_t)2) - 2;
            if (seqLen == 0) {
                ssWriter.writeData(ssData, 0, key, thread_idx);
                continue;
            }

            // Read AA sequence
            unsigned int aaId = aaReader.getId(key);
            const char *aaData = aaReader.getData(aaId, thread_idx);

            // Read CA coordinates
            unsigned int caId = caReader.getId(key);
            const char *caData = caReader.getData(caId, thread_idx);
            size_t caLen = caReader.getEntryLen(caId);
            float *caFloats = coords.read(caData, seqLen, caLen);

            // Convert flat floats to Vec3 arrays
            caVec.resize(seqLen);
            nVec.resize(seqLen);
            cVec.resize(seqLen);
            cbVec.resize(seqLen);
            amiVec.resize(seqLen);
            double nan = std::numeric_limits<double>::quiet_NaN();
            for (size_t j = 0; j < seqLen; j++) {
                caVec[j] = Vec3(caFloats[j], caFloats[seqLen + j], caFloats[2 * seqLen + j]);
                nVec[j] = Vec3(nan, nan, nan);
                cVec[j] = Vec3(nan, nan, nan);
                cbVec[j] = Vec3(nan, nan, nan);
                // Convert AA character to single-letter code for PULCHRA
                amiVec[j] = aaData[j];
            }

            // Reconstruct backbone from CA coordinates
            pulchra.rebuildBackbone(caVec.data(), nVec.data(), cVec.data(), amiVec.data(), seqLen);

            // Predict 12st states
            char *states12st = structureTo12St.structure2states(caVec.data(), nVec.data(),
                                                                cVec.data(), cbVec.data(), seqLen);

            // Encode combined 3Di + 12st
            result.clear();
            for (size_t j = 0; j < seqLen; j++) {
                int state3di = mat.aa2num[static_cast<unsigned char>(ssData[j])];
                result.push_back(static_cast<char>(state3di * Alphabet12St::STATE_CNT + states12st[j]));
            }
            result.push_back('\n');

            ssWriter.writeData(result.c_str(), result.size(), key, thread_idx);
        }
    }

    ssWriter.close();
    ssOldReader.close();
    caReader.close();
    aaReader.close();

    Debug(Debug::INFO) << "Database updated successfully. Old _ss saved as _ss_old.\n";
    return EXIT_SUCCESS;
}
