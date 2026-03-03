#ifndef LULESH_EOS_H
#define LULESH_EOS_H

#include "lulesh.h"

void ApplyMaterialPropertiesForElems(Domain& domain, Real_t vnew[]);
void UpdateVolumesForElems(Domain &domain, Real_t *vnew, Real_t v_cut, Index_t length);

#endif // LULESH_EOS_H
