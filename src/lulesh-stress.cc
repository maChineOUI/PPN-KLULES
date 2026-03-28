#include <cstdlib>
#include "lulesh-stress.h"
#include "lulesh-geometry.h"

/******************************************/

KOKKOS_INLINE_FUNCTION
static void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t fx[], Real_t fy[], Real_t fz[] )
{
   for (Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i] );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

static inline
void IntegrateStressForElems( Domain& domain,
                              Kokkos::View<Real_t*> determ,
                              Index_t numElem, Index_t numNode)
{
   Index_t numElem8 = numElem * 8 ;

   // Scatter buffers: each element writes to its own 8-slot block,
   // eliminating node-level write conflicts between parallel threads.
   // Using Kokkos::View so the buffer lives in the execution space memory
   // (device memory on GPU, host memory on CPU).
   Kokkos::View<Real_t*> fx_elem("fx_elem", numElem8) ;
   Kokkos::View<Real_t*> fy_elem("fy_elem", numElem8) ;
   Kokkos::View<Real_t*> fz_elem("fz_elem", numElem8) ;

   // Extract Views for KOKKOS_LAMBDA
   auto nodelist_v = domain.m_conn.m_nodelist ;
   auto x_v        = domain.m_nodes.m_x ;
   auto y_v        = domain.m_nodes.m_y ;
   auto z_v        = domain.m_nodes.m_z ;
   auto p_v        = domain.m_elems.m_p ;
   auto q_v        = domain.m_elems.m_q ;

   Kokkos::parallel_for("IntegrateStressForElems_scatter", numElem,
                        KOKKOS_LAMBDA(Index_t k) {
      const Index_t* elemToNode = nodelist_v.data() + 8*k ;
      Real_t B[3][8] ;
      Real_t x_local[8], y_local[8], z_local[8] ;
      Real_t detJ_k ;

      // Inline CollectDomainNodesToElemNodes
      for (Index_t n = 0; n < 8; ++n) {
         Index_t gn = elemToNode[n] ;
         x_local[n] = x_v(gn) ;
         y_local[n] = y_v(gn) ;
         z_local[n] = z_v(gn) ;
      }

      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ_k) ;
      determ(k) = detJ_k ;

      CalcElemNodeNormals(B[0], B[1], B[2], x_local, y_local, z_local) ;

      /* Opt-8: inline InitStressTermsForElems */
      Real_t sig = -p_v(k) - q_v(k) ;

      Real_t fx_local[8], fy_local[8], fz_local[8] ;
      SumElemStressesToNodeForces(B, sig, sig, sig, fx_local, fy_local, fz_local) ;

      Index_t base = k * 8 ;
      for (Index_t ni = 0; ni < 8; ++ni) {
         fx_elem(base + ni) = fx_local[ni] ;
         fy_elem(base + ni) = fy_local[ni] ;
         fz_elem(base + ni) = fz_local[ni] ;
      }
   });

   // Gather phase: accumulate per-corner forces into node forces
   auto nodeElemStart_v      = domain.m_conn.m_nodeElemStart ;
   auto nodeElemCornerList_v = domain.m_conn.m_nodeElemCornerList ;
   auto fx_v = domain.m_nodes.m_fx ;
   auto fy_v = domain.m_nodes.m_fy ;
   auto fz_v = domain.m_nodes.m_fz ;

   Kokkos::parallel_for("IntegrateStressForElems_gather", numNode,
                        KOKKOS_LAMBDA(Index_t gnode) {
      Index_t start = nodeElemStart_v(gnode) ;
      Index_t end   = nodeElemStart_v(gnode + 1) ;
      Real_t fx_tmp = Real_t(0.0) ;
      Real_t fy_tmp = Real_t(0.0) ;
      Real_t fz_tmp = Real_t(0.0) ;
      for (Index_t i = start; i < end; ++i) {
         Index_t corner = nodeElemCornerList_v(i) ;
         fx_tmp += fx_elem(corner) ;
         fy_tmp += fy_elem(corner) ;
         fz_tmp += fz_elem(corner) ;
      }
      fx_v(gnode) = fx_tmp ;
      fy_v(gnode) = fy_tmp ;
      fz_v(gnode) = fz_tmp ;
   });
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                              Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz)
{
   Real_t hxx[4];
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

static inline
void CalcHourglassControlForElems(Domain& domain,
                                  Kokkos::View<Real_t*> determ, Real_t hgcoef)
{
   Index_t numElem  = domain.numElem() ;
   Index_t numElem8 = numElem * 8 ;

   // Hourglass gamma matrix (Flanagan-Belytschko)
   const Real_t gamma[4][8] = {
      { Real_t( 1.), Real_t( 1.), Real_t(-1.), Real_t(-1.),
        Real_t(-1.), Real_t(-1.), Real_t( 1.), Real_t( 1.) },
      { Real_t( 1.), Real_t(-1.), Real_t(-1.), Real_t( 1.),
        Real_t(-1.), Real_t( 1.), Real_t( 1.), Real_t(-1.) },
      { Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.),
        Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) },
      { Real_t(-1.), Real_t( 1.), Real_t(-1.), Real_t( 1.),
        Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) }
   } ;

   // Extract Views
   auto nodelist_v = domain.m_conn.m_nodelist ;
   auto x_v        = domain.m_nodes.m_x ;
   auto y_v        = domain.m_nodes.m_y ;
   auto z_v        = domain.m_nodes.m_z ;
   auto xd_v       = domain.m_nodes.m_xd ;
   auto yd_v       = domain.m_nodes.m_yd ;
   auto zd_v       = domain.m_nodes.m_zd ;
   auto v_v        = domain.m_elems.m_v ;
   auto volo_v     = domain.m_elems.m_volo ;
   auto ss_v       = domain.m_elems.m_ss ;
   auto elemMass_v = domain.m_elems.m_elemMass ;

   if (hgcoef > Real_t(0.)) {
      // Scatter buffers for hourglass forces
      Kokkos::View<Real_t*> fx_elem("hg_fx_elem", numElem8) ;
      Kokkos::View<Real_t*> fy_elem("hg_fy_elem", numElem8) ;
      Kokkos::View<Real_t*> fz_elem("hg_fz_elem", numElem8) ;

      /* Fused loop: volume derivatives + FB hourglass scatter */
      Kokkos::parallel_for("CalcHourglassControlForElems", numElem,
                           KOKKOS_LAMBDA(Index_t i) {
         Real_t x1[8], y1[8], z1[8] ;
         Real_t pfx[8], pfy[8], pfz[8] ;

         const Index_t* elemToNode = nodelist_v.data() + 8*i ;

         // Inline CollectDomainNodesToElemNodes
         for (Index_t n = 0; n < 8; ++n) {
            Index_t gn = elemToNode[n] ;
            x1[n] = x_v(gn) ;
            y1[n] = y_v(gn) ;
            z1[n] = z_v(gn) ;
         }

         CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1) ;

         determ(i) = volo_v(i) * v_v(i) ;

         if (v_v(i) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive volume in CalcHourglassControlForElems") ;

         Real_t hgfx[8], hgfy[8], hgfz[8] ;
         Real_t hourgam[8][4] ;
         Real_t xd1[8], yd1[8], zd1[8] ;

         Index_t i3    = 8 * i ;
         Real_t volinv = Real_t(1.0) / determ(i) ;

         for (Index_t i1 = 0; i1 < 4; ++i1) {
            Real_t hourmodx =
               x1[0]*gamma[i1][0] + x1[1]*gamma[i1][1] +
               x1[2]*gamma[i1][2] + x1[3]*gamma[i1][3] +
               x1[4]*gamma[i1][4] + x1[5]*gamma[i1][5] +
               x1[6]*gamma[i1][6] + x1[7]*gamma[i1][7] ;
            Real_t hourmody =
               y1[0]*gamma[i1][0] + y1[1]*gamma[i1][1] +
               y1[2]*gamma[i1][2] + y1[3]*gamma[i1][3] +
               y1[4]*gamma[i1][4] + y1[5]*gamma[i1][5] +
               y1[6]*gamma[i1][6] + y1[7]*gamma[i1][7] ;
            Real_t hourmodz =
               z1[0]*gamma[i1][0] + z1[1]*gamma[i1][1] +
               z1[2]*gamma[i1][2] + z1[3]*gamma[i1][3] +
               z1[4]*gamma[i1][4] + z1[5]*gamma[i1][5] +
               z1[6]*gamma[i1][6] + z1[7]*gamma[i1][7] ;

            hourgam[0][i1] = gamma[i1][0] - volinv*(pfx[0]*hourmodx + pfy[0]*hourmody + pfz[0]*hourmodz);
            hourgam[1][i1] = gamma[i1][1] - volinv*(pfx[1]*hourmodx + pfy[1]*hourmody + pfz[1]*hourmodz);
            hourgam[2][i1] = gamma[i1][2] - volinv*(pfx[2]*hourmodx + pfy[2]*hourmody + pfz[2]*hourmodz);
            hourgam[3][i1] = gamma[i1][3] - volinv*(pfx[3]*hourmodx + pfy[3]*hourmody + pfz[3]*hourmodz);
            hourgam[4][i1] = gamma[i1][4] - volinv*(pfx[4]*hourmodx + pfy[4]*hourmody + pfz[4]*hourmodz);
            hourgam[5][i1] = gamma[i1][5] - volinv*(pfx[5]*hourmodx + pfy[5]*hourmody + pfz[5]*hourmodz);
            hourgam[6][i1] = gamma[i1][6] - volinv*(pfx[6]*hourmodx + pfy[6]*hourmody + pfz[6]*hourmodz);
            hourgam[7][i1] = gamma[i1][7] - volinv*(pfx[7]*hourmodx + pfy[7]*hourmody + pfz[7]*hourmodz);
         }

         Real_t ss1      = ss_v(i) ;
         Real_t mass1    = elemMass_v(i) ;
         Real_t volume13 = Kokkos::cbrt(determ(i)) ;

         for (Index_t n = 0; n < 8; ++n) {
            Index_t gn = elemToNode[n] ;
            xd1[n] = xd_v(gn) ;
            yd1[n] = yd_v(gn) ;
            zd1[n] = zd_v(gn) ;
         }

         Real_t coefficient = -hgcoef * Real_t(0.01) * ss1 * mass1 / volume13 ;

         CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient, hgfx, hgfy, hgfz) ;

         fx_elem(i3  ) = hgfx[0]; fy_elem(i3  ) = hgfy[0]; fz_elem(i3  ) = hgfz[0];
         fx_elem(i3+1) = hgfx[1]; fy_elem(i3+1) = hgfy[1]; fz_elem(i3+1) = hgfz[1];
         fx_elem(i3+2) = hgfx[2]; fy_elem(i3+2) = hgfy[2]; fz_elem(i3+2) = hgfz[2];
         fx_elem(i3+3) = hgfx[3]; fy_elem(i3+3) = hgfy[3]; fz_elem(i3+3) = hgfz[3];
         fx_elem(i3+4) = hgfx[4]; fy_elem(i3+4) = hgfy[4]; fz_elem(i3+4) = hgfz[4];
         fx_elem(i3+5) = hgfx[5]; fy_elem(i3+5) = hgfy[5]; fz_elem(i3+5) = hgfz[5];
         fx_elem(i3+6) = hgfx[6]; fy_elem(i3+6) = hgfy[6]; fz_elem(i3+6) = hgfz[6];
         fx_elem(i3+7) = hgfx[7]; fy_elem(i3+7) = hgfy[7]; fz_elem(i3+7) = hgfz[7];
      });

      // Gather: add hourglass forces to node forces
      auto nodeElemStart_v      = domain.m_conn.m_nodeElemStart ;
      auto nodeElemCornerList_v = domain.m_conn.m_nodeElemCornerList ;
      auto fx_v = domain.m_nodes.m_fx ;
      auto fy_v = domain.m_nodes.m_fy ;
      auto fz_v = domain.m_nodes.m_fz ;

      Kokkos::parallel_for("CalcFBHourglassForceForElems_gather", domain.numNode(),
                           KOKKOS_LAMBDA(Index_t gnode) {
         Index_t start = nodeElemStart_v(gnode) ;
         Index_t end   = nodeElemStart_v(gnode + 1) ;
         Real_t fx_tmp = Real_t(0.0) ;
         Real_t fy_tmp = Real_t(0.0) ;
         Real_t fz_tmp = Real_t(0.0) ;
         for (Index_t j = start; j < end; ++j) {
            Index_t corner = nodeElemCornerList_v(j) ;
            fx_tmp += fx_elem(corner) ;
            fy_tmp += fy_elem(corner) ;
            fz_tmp += fz_elem(corner) ;
         }
         fx_v(gnode) += fx_tmp ;
         fy_v(gnode) += fy_tmp ;
         fz_v(gnode) += fz_tmp ;
      });

   } else {
      // hgcoef == 0: still need to fill determ (used by volume check)
      Kokkos::parallel_for("CalcHourglassControlForElems_determ", numElem,
                           KOKKOS_LAMBDA(Index_t i) {
         determ(i) = volo_v(i) * v_v(i) ;
         if (v_v(i) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive volume in CalcHourglassControlForElems") ;
      });
   }
}

/******************************************/

void CalcVolumeForceForElems(Domain& domain)
{
   Index_t numElem = domain.numElem() ;
   if (numElem != 0) {
      Real_t hgcoef = domain.hgcoef() ;

      // determ: Jacobian determinant per element — Kokkos::View so GPU kernels
      // (IntegrateStress scatter/check, CalcHourglass) can access it.
      Kokkos::View<Real_t*> determ("determ", numElem) ;

      IntegrateStressForElems(domain, determ, numElem, domain.numNode()) ;

      // Check for negative element volume
      Kokkos::parallel_for("CalcVolumeForceForElems_check", numElem,
                           KOKKOS_LAMBDA(Index_t k) {
         if (determ(k) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive Jacobian determinant") ;
      });

      CalcHourglassControlForElems(domain, determ, hgcoef) ;
   }
}
