#pragma once

#include "lulesh.h"

void CalcQForElems(Domain& domain, Kokkos::View<Real_t*> vnew);
