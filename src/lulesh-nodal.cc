#include "lulesh-nodal.h"
#include "lulesh-stress.h"

#if _OPENMP
# include <omp.h>
#endif

/******************************************/

static inline void CalcForceForNodes(Domain& domain)
{
  Index_t numNode = domain.numNode() ;

#pragma omp parallel for firstprivate(numNode)
  for (Index_t i=0; i<numNode; ++i) {
     domain.fx(i) = Real_t(0.0) ;
     domain.fy(i) = Real_t(0.0) ;
     domain.fz(i) = Real_t(0.0) ;
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;
}

/******************************************/

static inline
void CalcAccelerationForNodes(Domain &domain, Index_t numNode)
{

#pragma omp parallel for firstprivate(numNode)
   for (Index_t i = 0; i < numNode; ++i) {
      domain.xdd(i) = domain.fx(i) / domain.nodalMass(i);
      domain.ydd(i) = domain.fy(i) / domain.nodalMass(i);
      domain.zdd(i) = domain.fz(i) / domain.nodalMass(i);
   }
}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
{
   Index_t size = domain.sizeX();
   Index_t numNodeBC = (size+1)*(size+1) ;

#pragma omp parallel
   {
      if (!domain.symmXempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(Index_t i=0 ; i<numNodeBC ; ++i)
            domain.xdd(domain.symmX(i)) = Real_t(0.0) ;
      }

      if (!domain.symmYempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(Index_t i=0 ; i<numNodeBC ; ++i)
            domain.ydd(domain.symmY(i)) = Real_t(0.0) ;
      }

      if (!domain.symmZempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(Index_t i=0 ; i<numNodeBC ; ++i)
            domain.zdd(domain.symmZ(i)) = Real_t(0.0) ;
      }
   }
}

/******************************************/

static inline
void CalcVelocityForNodes(Domain &domain, const Real_t dt, const Real_t u_cut,
                          Index_t numNode)
{

#pragma omp parallel for firstprivate(numNode)
   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = domain.xd(i) + domain.xdd(i) * dt ;
     if( std::fabs(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
     domain.xd(i) = xdtmp ;

     ydtmp = domain.yd(i) + domain.ydd(i) * dt ;
     if( std::fabs(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
     domain.yd(i) = ydtmp ;

     zdtmp = domain.zd(i) + domain.zdd(i) * dt ;
     if( std::fabs(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
     domain.zd(i) = zdtmp ;
   }
}

/******************************************/

static inline
void CalcPositionForNodes(Domain &domain, const Real_t dt, Index_t numNode)
{
#pragma omp parallel for firstprivate(numNode)
   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
     domain.x(i) += domain.xd(i) * dt ;
     domain.y(i) += domain.yd(i) * dt ;
     domain.z(i) += domain.zd(i) * dt ;
   }
}

/******************************************/

void LagrangeNodal(Domain& domain)
{
   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

   CalcAccelerationForNodes(domain, domain.numNode());

   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityForNodes( domain, delt, u_cut, domain.numNode()) ;

   CalcPositionForNodes( domain, delt, domain.numNode() );

  return;
}
