#include "lulesh-integration.h"
#include "lulesh-comm.h"
#include "lulesh-kinematics.h"
#include "lulesh-viscosity.h"
#include "lulesh-eos.h"
#include "lulesh-nodal.h"
#include "lulesh-timestep.h"

/******************************************/

static inline
void LagrangeElements(Domain& domain, Index_t numElem)
{
  Real_t *vnew = Allocate<Real_t>(numElem) ;  /* new relative vol -- temp */

  CalcLagrangeElements(domain, vnew) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain, vnew) ;

  ApplyMaterialPropertiesForElems(domain, vnew) ;

  UpdateVolumesForElems(domain, vnew,
                        domain.v_cut(), numElem) ;

  Release(&vnew);
}

/******************************************/

void LagrangeLeapFrog(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_LATE
   Domain_member fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);


#ifdef SEDOV_SYNC_POS_VEL_LATE
#endif

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain.numElem());

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_LATE
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
#endif
#endif

   CalcTimeConstraintsForElems(domain);

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_LATE
   CommSyncPosVel(domain) ;
#endif
#endif
}
