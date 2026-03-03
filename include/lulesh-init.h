#ifndef LULESH_INIT_H
#define LULESH_INIT_H

#include "lulesh.h"

void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);

#endif // LULESH_INIT_H
