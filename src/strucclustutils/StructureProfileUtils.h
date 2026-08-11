#pragma once

#include <string>

#include "LocalParameters.h"

struct ResolvedStructureProfileDbs {
    std::string ssInputDb;
    std::string ssTargetDb;
    std::string ss12InputDb;
    std::string ss12TargetDb;
    bool removeSplitTmp;
};

ResolvedStructureProfileDbs resolveStructureProfileDbs(LocalParameters &par,
                                                       const std::string &db2NoIndexName,
                                                       bool disableSs12Profile);

std::string createAaProfileParameters(LocalParameters &par);
std::string create3DiProfileParameters(LocalParameters &par);
std::string create12StateProfileParameters(LocalParameters &par);
