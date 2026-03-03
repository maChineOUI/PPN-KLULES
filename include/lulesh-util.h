#ifndef LULESH_UTIL_H
#define LULESH_UTIL_H

#include "lulesh.h"

void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);

#endif // LULESH_UTIL_H
