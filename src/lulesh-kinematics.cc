#include "lulesh-kinematics.h"
#include "lulesh-geometry.h"

/******************************************/

static inline
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

void CalcLagrangeElements(Domain& domain, Real_t* vnew)
{
   Index_t numElem = domain.numElem() ;
   if (numElem > 0) {
      const Real_t deltatime = domain.deltatime() ;

      domain.AllocateStrains(numElem);

      /* Opt-9: Fused kinematics + Lagrange — D[0]/D[1]/D[2] stay as stack
         scalars, eliminating the intermediate write/read of domain.dxx/dyy/dzz
         and the barrier between the two former parallel_for kernels. */
      Kokkos::parallel_for("CalcKinematicsAndLagrange", numElem,
                           [&](Index_t k) {
         Real_t B[3][8] ;
         Real_t D[6] ;
         Real_t x_local[8], y_local[8], z_local[8] ;
         Real_t xd_local[8], yd_local[8], zd_local[8] ;
         Real_t detJ = Real_t(0.0) ;

         const Index_t* const elemToNode = domain.nodelist(k) ;

         CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

         Real_t volume = CalcElemVolume(x_local, y_local, z_local) ;
         Real_t relativeVolume = volume / domain.volo(k) ;
         vnew[k] = relativeVolume ;
         domain.delv(k) = relativeVolume - domain.v(k) ;

         domain.arealg(k) = CalcElemCharacteristicLength(x_local, y_local, z_local, volume) ;

         for (Index_t lnode=0 ; lnode<8 ; ++lnode) {
            Index_t gnode = elemToNode[lnode] ;
            xd_local[lnode] = domain.xd(gnode) ;
            yd_local[lnode] = domain.yd(gnode) ;
            zd_local[lnode] = domain.zd(gnode) ;
         }

         Real_t dt2 = Real_t(0.5) * deltatime ;
         for (Index_t j=0 ; j<8 ; ++j) {
            x_local[j] -= dt2 * xd_local[j] ;
            y_local[j] -= dt2 * yd_local[j] ;
            z_local[j] -= dt2 * zd_local[j] ;
         }

         CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ) ;
         CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D) ;

         // Lagrange part: vdov + deviatoric strain
         Real_t vdov      = D[0] + D[1] + D[2] ;
         Real_t vdovthird = vdov / Real_t(3.0) ;
         domain.vdov(k) = vdov ;
         domain.dxx(k)  = D[0] - vdovthird ;
         domain.dyy(k)  = D[1] - vdovthird ;
         domain.dzz(k)  = D[2] - vdovthird ;

         if (vnew[k] <= Real_t(0.0))
            exit(VolumeError) ;
      });

      domain.DeallocateStrains();
   }
}
