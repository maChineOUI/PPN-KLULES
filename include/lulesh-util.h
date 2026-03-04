#pragma once

#include "lulesh.h"

struct cmdLineOpts {
   Int_t its;      // -i
   Int_t nx;       // -s
   Int_t numReg;   // -r
   Int_t numFiles; // -f
   Int_t showProg; // -p
   Int_t quiet;    // -q
   Int_t cost;     // -c
   Int_t balance;  // -b
};

void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);
