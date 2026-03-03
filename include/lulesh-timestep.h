#ifndef LULESH_TIMESTEP_H
#define LULESH_TIMESTEP_H

#include "lulesh.h"

void TimeIncrement(Domain& domain);
void CalcTimeConstraintsForElems(Domain& domain);

#endif // LULESH_TIMESTEP_H
