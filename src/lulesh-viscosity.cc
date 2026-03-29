#include "lulesh-viscosity.h"

/******************************************/

static inline
void CalcMonotonicQGradientsForElems(Domain& domain, Kokkos::View<Real_t*> vnew)
{
   Index_t numElem = domain.numElem();

   // Extract Views for KOKKOS_LAMBDA ([=] capture)
   auto nodelist_v  = domain.m_conn.m_nodelist ;
   auto x_v         = domain.m_nodes.m_x ;
   auto y_v         = domain.m_nodes.m_y ;
   auto z_v         = domain.m_nodes.m_z ;
   auto xd_v        = domain.m_nodes.m_xd ;
   auto yd_v        = domain.m_nodes.m_yd ;
   auto zd_v        = domain.m_nodes.m_zd ;
   auto volo_v      = domain.m_elems.m_volo ;
   auto delx_zeta_v = domain.m_elems.m_delx_zeta ;
   auto delx_xi_v   = domain.m_elems.m_delx_xi ;
   auto delx_eta_v  = domain.m_elems.m_delx_eta ;
   auto delv_zeta_v = domain.m_elems.m_delv_zeta ;
   auto delv_xi_v   = domain.m_elems.m_delv_xi ;
   auto delv_eta_v  = domain.m_elems.m_delv_eta ;

   Kokkos::parallel_for("CalcMonotonicQGradientsForElems", numElem,
                        KOKKOS_LAMBDA(Index_t i) {
      const Real_t ptiny = Real_t(1.e-36) ;
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      const Index_t* elemToNode = nodelist_v.data() + 8*i ;
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = x_v(n0) ; Real_t x1 = x_v(n1) ;
      Real_t x2 = x_v(n2) ; Real_t x3 = x_v(n3) ;
      Real_t x4 = x_v(n4) ; Real_t x5 = x_v(n5) ;
      Real_t x6 = x_v(n6) ; Real_t x7 = x_v(n7) ;

      Real_t y0 = y_v(n0) ; Real_t y1 = y_v(n1) ;
      Real_t y2 = y_v(n2) ; Real_t y3 = y_v(n3) ;
      Real_t y4 = y_v(n4) ; Real_t y5 = y_v(n5) ;
      Real_t y6 = y_v(n6) ; Real_t y7 = y_v(n7) ;

      Real_t z0 = z_v(n0) ; Real_t z1 = z_v(n1) ;
      Real_t z2 = z_v(n2) ; Real_t z3 = z_v(n3) ;
      Real_t z4 = z_v(n4) ; Real_t z5 = z_v(n5) ;
      Real_t z6 = z_v(n6) ; Real_t z7 = z_v(n7) ;

      Real_t xv0 = xd_v(n0) ; Real_t xv1 = xd_v(n1) ;
      Real_t xv2 = xd_v(n2) ; Real_t xv3 = xd_v(n3) ;
      Real_t xv4 = xd_v(n4) ; Real_t xv5 = xd_v(n5) ;
      Real_t xv6 = xd_v(n6) ; Real_t xv7 = xd_v(n7) ;

      Real_t yv0 = yd_v(n0) ; Real_t yv1 = yd_v(n1) ;
      Real_t yv2 = yd_v(n2) ; Real_t yv3 = yd_v(n3) ;
      Real_t yv4 = yd_v(n4) ; Real_t yv5 = yd_v(n5) ;
      Real_t yv6 = yd_v(n6) ; Real_t yv7 = yd_v(n7) ;

      Real_t zv0 = zd_v(n0) ; Real_t zv1 = zd_v(n1) ;
      Real_t zv2 = zd_v(n2) ; Real_t zv3 = zd_v(n3) ;
      Real_t zv4 = zd_v(n4) ; Real_t zv5 = zd_v(n5) ;
      Real_t zv6 = zd_v(n6) ; Real_t zv7 = zd_v(n7) ;

      Real_t vol = volo_v(i)*vnew(i) ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
      Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
      Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

      Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
      Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
      Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

      Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
      Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
      Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

      /* find delvk and delxk ( i cross j ) */
      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      delx_zeta_v(i) = vol / Kokkos::sqrt(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ; ay *= norm ; az *= norm ;

      dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
      dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
      dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

      delv_zeta_v(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */
      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      delx_xi_v(i) = vol / Kokkos::sqrt(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ; ay *= norm ; az *= norm ;

      dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
      dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
      dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

      delv_xi_v(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */
      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      delx_eta_v(i) = vol / Kokkos::sqrt(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ; ay *= norm ; az *= norm ;

      dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
      dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
      dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

      delv_eta_v(i) = ax*dxv + ay*dyv + az*dzv ;
   });
}

/******************************************/

static inline
void CalcMonotonicQForElems(Domain& domain, Kokkos::View<Real_t*> vnew)
{
   /* Opt-10: flatten 11 per-region parallel_for into one over all numElem. */
   const Real_t ptiny               = Real_t(1.e-36) ;
   const Real_t monoq_limiter_mult  = domain.monoq_limiter_mult() ;
   const Real_t monoq_max_slope     = domain.monoq_max_slope() ;
   const Real_t qlc_monoq           = domain.qlc_monoq() ;
   const Real_t qqc_monoq           = domain.qqc_monoq() ;
   Index_t numElem = domain.numElem() ;

   // Extract Views
   auto elemBC_v   = domain.m_conn.m_elemBC ;
   auto lxim_v     = domain.m_conn.m_lxim ;
   auto lxip_v     = domain.m_conn.m_lxip ;
   auto letam_v    = domain.m_conn.m_letam ;
   auto letap_v    = domain.m_conn.m_letap ;
   auto lzetam_v   = domain.m_conn.m_lzetam ;
   auto lzetap_v   = domain.m_conn.m_lzetap ;
   auto delv_xi_v   = domain.m_elems.m_delv_xi ;
   auto delv_eta_v  = domain.m_elems.m_delv_eta ;
   auto delv_zeta_v = domain.m_elems.m_delv_zeta ;
   auto delx_xi_v   = domain.m_elems.m_delx_xi ;
   auto delx_eta_v  = domain.m_elems.m_delx_eta ;
   auto delx_zeta_v = domain.m_elems.m_delx_zeta ;
   auto vdov_v      = domain.m_elems.m_vdov ;
   auto elemMass_v  = domain.m_elems.m_elemMass ;
   auto volo_v      = domain.m_elems.m_volo ;
   auto qq_v        = domain.m_elems.m_qq ;
   auto ql_v        = domain.m_elems.m_ql ;

   Kokkos::parallel_for("CalcMonotonicQForElems", numElem,
                        KOKKOS_LAMBDA(Index_t i) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Int_t bcMask = elemBC_v(i) ;
      Real_t delvm = 0.0, delvp = 0.0 ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / (delv_xi_v(i) + ptiny) ;

      switch (bcMask & XI_M) {
         case XI_M_COMM: /* needs comm data */
         case 0:         delvm = delv_xi_v(lxim_v(i)) ; break ;
         case XI_M_SYMM: delvm = delv_xi_v(i) ;         break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;           break ;
         default:        Kokkos::abort("CalcMonotonicQForElems: bad XI_M bcMask");
                         delvm = 0; break;
      }
      switch (bcMask & XI_P) {
         case XI_P_COMM: /* needs comm data */
         case 0:         delvp = delv_xi_v(lxip_v(i)) ; break ;
         case XI_P_SYMM: delvp = delv_xi_v(i) ;         break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;           break ;
         default:        Kokkos::abort("CalcMonotonicQForElems: bad XI_P bcMask");
                         delvp = 0; break;
      }
      delvm *= norm ; delvp *= norm ;
      phixi = Real_t(.5) * (delvm + delvp) ;
      delvm *= monoq_limiter_mult ; delvp *= monoq_limiter_mult ;
      if (delvm < phixi) phixi = delvm ;
      if (delvp < phixi) phixi = delvp ;
      if (phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if (phixi > monoq_max_slope) phixi = monoq_max_slope ;

      /*  phieta     */
      norm = Real_t(1.) / (delv_eta_v(i) + ptiny) ;

      switch (bcMask & ETA_M) {
         case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = delv_eta_v(letam_v(i)) ; break ;
         case ETA_M_SYMM: delvm = delv_eta_v(i) ;          break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;             break ;
         default:         Kokkos::abort("CalcMonotonicQForElems: bad ETA_M bcMask");
                          delvm = 0; break;
      }
      switch (bcMask & ETA_P) {
         case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = delv_eta_v(letap_v(i)) ; break ;
         case ETA_P_SYMM: delvp = delv_eta_v(i) ;          break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;             break ;
         default:         Kokkos::abort("CalcMonotonicQForElems: bad ETA_P bcMask");
                          delvp = 0; break;
      }
      delvm *= norm ; delvp *= norm ;
      phieta = Real_t(.5) * (delvm + delvp) ;
      delvm *= monoq_limiter_mult ; delvp *= monoq_limiter_mult ;
      if (delvm  < phieta) phieta = delvm ;
      if (delvp  < phieta) phieta = delvp ;
      if (phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if (phieta > monoq_max_slope) phieta = monoq_max_slope ;

      /*  phizeta     */
      norm = Real_t(1.) / (delv_zeta_v(i) + ptiny) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = delv_zeta_v(lzetam_v(i)) ; break ;
         case ZETA_M_SYMM: delvm = delv_zeta_v(i) ;           break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;               break ;
         default:          Kokkos::abort("CalcMonotonicQForElems: bad ZETA_M bcMask");
                           delvm = 0; break;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = delv_zeta_v(lzetap_v(i)) ; break ;
         case ZETA_P_SYMM: delvp = delv_zeta_v(i) ;           break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;               break ;
         default:          Kokkos::abort("CalcMonotonicQForElems: bad ZETA_P bcMask");
                           delvp = 0; break;
      }
      delvm *= norm ; delvp *= norm ;
      phizeta = Real_t(.5) * (delvm + delvp) ;
      delvm *= monoq_limiter_mult ; delvp *= monoq_limiter_mult ;
      if (delvm   < phizeta) phizeta = delvm ;
      if (delvp   < phizeta) phizeta = delvp ;
      if (phizeta < Real_t(0.)) phizeta = Real_t(0.) ;
      if (phizeta > monoq_max_slope) phizeta = monoq_max_slope ;

      /* Remove length scale */
      if (vdov_v(i) > Real_t(0.)) {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = delv_xi_v(i)   * delx_xi_v(i) ;
         Real_t delvxeta  = delv_eta_v(i)  * delx_eta_v(i) ;
         Real_t delvxzeta = delv_zeta_v(i) * delx_zeta_v(i) ;

         if (delvxxi   > Real_t(0.)) delvxxi   = Real_t(0.) ;
         if (delvxeta  > Real_t(0.)) delvxeta  = Real_t(0.) ;
         if (delvxzeta > Real_t(0.)) delvxzeta = Real_t(0.) ;

         Real_t rho = elemMass_v(i) / (volo_v(i) * vnew(i)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      qq_v(i) = qquad ;
      ql_v(i) = qlin  ;
   }) ;
}

/******************************************/

void CalcQForElems(Domain& domain, Kokkos::View<Real_t*> vnew)
{
   Index_t numElem = domain.numElem() ;

   if (numElem != 0) {
      Int_t allElem = numElem +          /* local elem */
            2*domain.sizeX()*domain.sizeY() + /* plane ghosts */
            2*domain.sizeX()*domain.sizeZ() + /* row ghosts */
            2*domain.sizeY()*domain.sizeZ() ; /* col ghosts */

      domain.AllocateGradients(numElem, allElem);

      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems(domain, vnew);

      CalcMonotonicQForElems(domain, vnew) ;

      // Free up memory
      domain.DeallocateGradients();

      /* Don't allow excessive artificial viscosity.
         m_q lives in device memory (CudaSpace on GPU); use a device-side
         parallel_reduce instead of a host-side loop. */
      auto q_v = domain.m_elems.m_q ;
      const Real_t qstop = domain.qstop() ;
      Index_t exceeded = 0 ;
      Kokkos::parallel_reduce("QStopCheck", numElem,
         KOKKOS_LAMBDA(Index_t i, Index_t& flag) {
            if (q_v(i) > qstop) flag = 1 ;
         },
         Kokkos::Max<Index_t>(exceeded)) ;

      if (exceeded > 0) {
         exit(QStopError);
      }
   }
}
