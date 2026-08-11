#include <cassert>
#include <cstdlib>
#include <string>

#include "CommandCaller.h"
#include "FileUtil.h"
#include "LocalParameters.h"
#include "PrefilteringIndexReader.h"
#include "StructureProfileUtils.h"
#include "result2structprofile.sh.h"

int result2structprofile(int argc, const char **argv, const Command &command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);

    CommandCaller cmd;
    const std::string db2NoIndexName = PrefilteringIndexReader::dbPathWithoutIndex(par.db2);

    const char *disableSs12ProfileEnv = std::getenv("FOLDSEEK_DISABLE_SS12_PROFILE");
    const bool disableSs12Profile = (disableSs12ProfileEnv != NULL
                                     && disableSs12ProfileEnv[0] != '\0'
                                     && disableSs12ProfileEnv[0] != '0');
    const ResolvedStructureProfileDbs resolved = resolveStructureProfileDbs(par, db2NoIndexName, disableSs12Profile);

    if (FileUtil::fileExists((par.db1 + ".dbtype").c_str())) {
        cmd.addVariable("PROFILE_PAR", createAaProfileParameters(par).c_str());
    }

    if (FileUtil::fileExists((resolved.ssInputDb + ".dbtype").c_str())) {
        const std::string ssTargetDb = resolved.ssTargetDb + (db2NoIndexName != par.db2 ? ".idx" : "");
        cmd.addVariable("INPUT_SS_DB", resolved.ssInputDb.c_str());
        cmd.addVariable("TARGET_SS_DB", ssTargetDb.c_str());
        cmd.addVariable("PROFILE_SS_PAR", create3DiProfileParameters(par).c_str());
    }

    if (resolved.ss12InputDb.empty() == false && resolved.ss12TargetDb.empty() == false) {
        cmd.addVariable("INPUT_SS12_DB", resolved.ss12InputDb.c_str());
        cmd.addVariable("TARGET_SS12_DB", resolved.ss12TargetDb.c_str());
        cmd.addVariable("PROFILE_SS12_PAR", create12StateProfileParameters(par).c_str());
        cmd.addVariable("CREATE_SS12_PROFILE", "1");
        if (par.profileOutputMode == 1) {
            cmd.addVariable("PROFILE_SS12_OUTPUT_COLUMNS", "12");
            cmd.addVariable("PROFILE_SS12_OUTPUT_LABELS", "A C D E F G H I K L M N");
        }
    }

    if (resolved.removeSplitTmp) {
        cmd.addVariable("REMOVE_SPLIT_TMP", "1");
    }

    cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());

    const std::string program = par.db4 + ".sh";
    FileUtil::writeFile(program, result2structprofile_sh, result2structprofile_sh_len);
    cmd.execProgram(FileUtil::getRealPathFromSymLink(program).c_str(), par.filenames);

    assert(false);
    return EXIT_FAILURE;
}
