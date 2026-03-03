#include "lulesh-integration.h"
#include "lulesh-kinematics.h"
#include "lulesh-viscosity.h"
#include "lulesh-eos.h"
#include "lulesh-nodal.h"
#include "lulesh-timestep.h"

/******************************************/

static inline
void LagrangeElements(Domain& domain, Index_t numElem)
{
  std::vector<Real_t> vnew(numElem) ;  /* new relative vol -- temp */

  CalcLagrangeElements(domain, vnew.data()) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain, vnew.data()) ;

  ApplyMaterialPropertiesForElems(domain, vnew.data()) ;

  UpdateVolumesForElems(domain, vnew.data(),
                        domain.v_cut(), numElem) ;
}

/******************************************/

void LagrangeLeapFrog(Domain& domain)
{
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain.numElem());

   CalcTimeConstraintsForElems(domain);
}
