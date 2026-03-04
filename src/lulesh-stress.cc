#include "lulesh-stress.h"
#include "lulesh-geometry.h"

/******************************************/

static inline
void InitStressTermsForElems(Domain &domain,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                             Index_t numElem)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //

   Kokkos::parallel_for("InitStressTermsForElems", numElem,
                        [&](Index_t i) {
      sigxx[i] = sigyy[i] = sigzz[i] =  - domain.p(i) - domain.q(i) ;
   });
}

/******************************************/

static inline
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t fx[], Real_t fy[], Real_t fz[] )
{
   for(Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i]  );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

static inline
void IntegrateStressForElems( Domain &domain,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, Index_t numElem, Index_t numNode)
{
   Index_t numthreads = Kokkos::DefaultHostExecutionSpace().concurrency();

   Index_t numElem8 = numElem * 8 ;
   std::vector<Real_t> fx_elem, fy_elem, fz_elem ;

  if (numthreads > 1) {
     fx_elem.resize(numElem8) ;
     fy_elem.resize(numElem8) ;
     fz_elem.resize(numElem8) ;
  }

  // Extract raw pointers for lambda capture
  Real_t* fx_elem_ptr = numthreads > 1 ? fx_elem.data() : nullptr ;
  Real_t* fy_elem_ptr = numthreads > 1 ? fy_elem.data() : nullptr ;
  Real_t* fz_elem_ptr = numthreads > 1 ? fz_elem.data() : nullptr ;

  // loop over all elements
  Kokkos::parallel_for("IntegrateStressForElems_scatter", numElem,
                       [&](Index_t k) {
    const Index_t* const elemToNode = domain.nodelist(k);
    Real_t B[3][8] ;// shape function derivatives
    Real_t x_local[8] ;
    Real_t y_local[8] ;
    Real_t z_local[8] ;

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    if (numthreads > 1) {
       // Eliminate thread writing conflicts at the nodes by giving
       // each element its own copy to write to
       Real_t fx_local[8], fy_local[8], fz_local[8] ;
       SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                    fx_local, fy_local, fz_local ) ;
       for (Index_t ni = 0; ni < 8; ++ni) {
          fx_elem_ptr[k*8+ni] = fx_local[ni] ;
          fy_elem_ptr[k*8+ni] = fy_local[ni] ;
          fz_elem_ptr[k*8+ni] = fz_local[ni] ;
       }
    }
    else {
       Real_t fx_local[8] ;
       Real_t fy_local[8] ;
       Real_t fz_local[8] ;
       SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                    fx_local, fy_local, fz_local ) ;

       // copy nodal force contributions to global force arrray.
       for( Index_t lnode=0 ; lnode<8 ; ++lnode ) {
          Index_t gnode = elemToNode[lnode];
          domain.fx(gnode) += fx_local[lnode];
          domain.fy(gnode) += fy_local[lnode];
          domain.fz(gnode) += fz_local[lnode];
       }
    }
  });

  if (numthreads > 1) {
     // If threaded, then we need to copy the data out of the temporary
     // arrays used above into the final forces field
     Kokkos::parallel_for("IntegrateStressForElems_gather", numNode,
                          [&](Index_t gnode) {
        Index_t count = domain.nodeElemCount(gnode) ;
        Index_t *cornerList = domain.nodeElemCornerList(gnode) ;
        Real_t fx_tmp = Real_t(0.0) ;
        Real_t fy_tmp = Real_t(0.0) ;
        Real_t fz_tmp = Real_t(0.0) ;
        for (Index_t i=0 ; i < count ; ++i) {
           Index_t elem = cornerList[i] ;
           fx_tmp += fx_elem_ptr[elem] ;
           fy_tmp += fy_elem_ptr[elem] ;
           fz_tmp += fz_elem_ptr[elem] ;
        }
        domain.fx(gnode) = fx_tmp ;
        domain.fy(gnode) = fy_tmp ;
        domain.fz(gnode) = fz_tmp ;
     });
  }
}

/******************************************/

static inline
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Real_t hxx[4];
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

static inline
void CalcFBHourglassForceForElems( Domain &domain,
                                   Real_t *determ,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem,
                                   Index_t numNode)
{

   Index_t numthreads = Kokkos::DefaultHostExecutionSpace().concurrency();
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

   Index_t numElem8 = numElem * 8 ;

   std::vector<Real_t> fx_elem, fy_elem, fz_elem ;

   if(numthreads > 1) {
      fx_elem.resize(numElem8) ;
      fy_elem.resize(numElem8) ;
      fz_elem.resize(numElem8) ;
   }

   // Extract raw pointers for lambda capture
   Real_t* fx_elem_ptr = numthreads > 1 ? fx_elem.data() : nullptr ;
   Real_t* fy_elem_ptr = numthreads > 1 ? fy_elem.data() : nullptr ;
   Real_t* fz_elem_ptr = numthreads > 1 ? fz_elem.data() : nullptr ;

   Real_t  gamma[4][8];

   gamma[0][0] = Real_t( 1.);
   gamma[0][1] = Real_t( 1.);
   gamma[0][2] = Real_t(-1.);
   gamma[0][3] = Real_t(-1.);
   gamma[0][4] = Real_t(-1.);
   gamma[0][5] = Real_t(-1.);
   gamma[0][6] = Real_t( 1.);
   gamma[0][7] = Real_t( 1.);
   gamma[1][0] = Real_t( 1.);
   gamma[1][1] = Real_t(-1.);
   gamma[1][2] = Real_t(-1.);
   gamma[1][3] = Real_t( 1.);
   gamma[1][4] = Real_t(-1.);
   gamma[1][5] = Real_t( 1.);
   gamma[1][6] = Real_t( 1.);
   gamma[1][7] = Real_t(-1.);
   gamma[2][0] = Real_t( 1.);
   gamma[2][1] = Real_t(-1.);
   gamma[2][2] = Real_t( 1.);
   gamma[2][3] = Real_t(-1.);
   gamma[2][4] = Real_t( 1.);
   gamma[2][5] = Real_t(-1.);
   gamma[2][6] = Real_t( 1.);
   gamma[2][7] = Real_t(-1.);
   gamma[3][0] = Real_t(-1.);
   gamma[3][1] = Real_t( 1.);
   gamma[3][2] = Real_t(-1.);
   gamma[3][3] = Real_t( 1.);
   gamma[3][4] = Real_t( 1.);
   gamma[3][5] = Real_t(-1.);
   gamma[3][6] = Real_t( 1.);
   gamma[3][7] = Real_t(-1.);

/*************************************************/
/*    compute the hourglass modes */

   Kokkos::parallel_for("CalcFBHourglassForceForElems_scatter", numElem,
                        [&](Index_t i2) {
      Real_t hgfx[8], hgfy[8], hgfz[8] ;

      Real_t coefficient;

      Real_t hourgam[8][4];
      Real_t xd1[8], yd1[8], zd1[8] ;

      const Index_t *elemToNode = domain.nodelist(i2);
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam[0][i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam[1][i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam[2][i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam[3][i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam[4][i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam[5][i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam[6][i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam[7][i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=domain.ss(i2);
      mass1=domain.elemMass(i2);
      volume13=std::cbrt(determ[i2]);

      Index_t n0si2 = elemToNode[0];
      Index_t n1si2 = elemToNode[1];
      Index_t n2si2 = elemToNode[2];
      Index_t n3si2 = elemToNode[3];
      Index_t n4si2 = elemToNode[4];
      Index_t n5si2 = elemToNode[5];
      Index_t n6si2 = elemToNode[6];
      Index_t n7si2 = elemToNode[7];

      xd1[0] = domain.xd(n0si2);
      xd1[1] = domain.xd(n1si2);
      xd1[2] = domain.xd(n2si2);
      xd1[3] = domain.xd(n3si2);
      xd1[4] = domain.xd(n4si2);
      xd1[5] = domain.xd(n5si2);
      xd1[6] = domain.xd(n6si2);
      xd1[7] = domain.xd(n7si2);

      yd1[0] = domain.yd(n0si2);
      yd1[1] = domain.yd(n1si2);
      yd1[2] = domain.yd(n2si2);
      yd1[3] = domain.yd(n3si2);
      yd1[4] = domain.yd(n4si2);
      yd1[5] = domain.yd(n5si2);
      yd1[6] = domain.yd(n6si2);
      yd1[7] = domain.yd(n7si2);

      zd1[0] = domain.zd(n0si2);
      zd1[1] = domain.zd(n1si2);
      zd1[2] = domain.zd(n2si2);
      zd1[3] = domain.zd(n3si2);
      zd1[4] = domain.zd(n4si2);
      zd1[5] = domain.zd(n5si2);
      zd1[6] = domain.zd(n6si2);
      zd1[7] = domain.zd(n7si2);

      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
                      hourgam,
                      coefficient, hgfx, hgfy, hgfz);

      // With the threaded version, we write into local arrays per elem
      // so we don't have to worry about race conditions
      if (numthreads > 1) {
         fx_elem_ptr[i3  ] = hgfx[0]; fy_elem_ptr[i3  ] = hgfy[0]; fz_elem_ptr[i3  ] = hgfz[0];
         fx_elem_ptr[i3+1] = hgfx[1]; fy_elem_ptr[i3+1] = hgfy[1]; fz_elem_ptr[i3+1] = hgfz[1];
         fx_elem_ptr[i3+2] = hgfx[2]; fy_elem_ptr[i3+2] = hgfy[2]; fz_elem_ptr[i3+2] = hgfz[2];
         fx_elem_ptr[i3+3] = hgfx[3]; fy_elem_ptr[i3+3] = hgfy[3]; fz_elem_ptr[i3+3] = hgfz[3];
         fx_elem_ptr[i3+4] = hgfx[4]; fy_elem_ptr[i3+4] = hgfy[4]; fz_elem_ptr[i3+4] = hgfz[4];
         fx_elem_ptr[i3+5] = hgfx[5]; fy_elem_ptr[i3+5] = hgfy[5]; fz_elem_ptr[i3+5] = hgfz[5];
         fx_elem_ptr[i3+6] = hgfx[6]; fy_elem_ptr[i3+6] = hgfy[6]; fz_elem_ptr[i3+6] = hgfz[6];
         fx_elem_ptr[i3+7] = hgfx[7]; fy_elem_ptr[i3+7] = hgfy[7]; fz_elem_ptr[i3+7] = hgfz[7];
      }
      else {
         domain.fx(n0si2) += hgfx[0]; domain.fy(n0si2) += hgfy[0]; domain.fz(n0si2) += hgfz[0];
         domain.fx(n1si2) += hgfx[1]; domain.fy(n1si2) += hgfy[1]; domain.fz(n1si2) += hgfz[1];
         domain.fx(n2si2) += hgfx[2]; domain.fy(n2si2) += hgfy[2]; domain.fz(n2si2) += hgfz[2];
         domain.fx(n3si2) += hgfx[3]; domain.fy(n3si2) += hgfy[3]; domain.fz(n3si2) += hgfz[3];
         domain.fx(n4si2) += hgfx[4]; domain.fy(n4si2) += hgfy[4]; domain.fz(n4si2) += hgfz[4];
         domain.fx(n5si2) += hgfx[5]; domain.fy(n5si2) += hgfy[5]; domain.fz(n5si2) += hgfz[5];
         domain.fx(n6si2) += hgfx[6]; domain.fy(n6si2) += hgfy[6]; domain.fz(n6si2) += hgfz[6];
         domain.fx(n7si2) += hgfx[7]; domain.fy(n7si2) += hgfy[7]; domain.fz(n7si2) += hgfz[7];
      }
   });

   if (numthreads > 1) {
     // Collect the data from the local arrays into the final force arrays
      Kokkos::parallel_for("CalcFBHourglassForceForElems_gather", numNode,
                           [&](Index_t gnode) {
         Index_t count = domain.nodeElemCount(gnode) ;
         Index_t *cornerList = domain.nodeElemCornerList(gnode) ;
         Real_t fx_tmp = Real_t(0.0) ;
         Real_t fy_tmp = Real_t(0.0) ;
         Real_t fz_tmp = Real_t(0.0) ;
         for (Index_t i=0 ; i < count ; ++i) {
            Index_t elem = cornerList[i] ;
            fx_tmp += fx_elem_ptr[elem] ;
            fy_tmp += fy_elem_ptr[elem] ;
            fz_tmp += fz_elem_ptr[elem] ;
         }
         domain.fx(gnode) += fx_tmp ;
         domain.fy(gnode) += fy_tmp ;
         domain.fz(gnode) += fz_tmp ;
      });
   }
}

/******************************************/

static inline
void CalcHourglassControlForElems(Domain& domain,
                                  Real_t determ[], Real_t hgcoef)
{
   Index_t numElem = domain.numElem() ;
   Index_t numElem8 = numElem * 8 ;
   std::vector<Real_t> dvdx(numElem8) ;
   std::vector<Real_t> dvdy(numElem8) ;
   std::vector<Real_t> dvdz(numElem8) ;
   std::vector<Real_t> x8n(numElem8) ;
   std::vector<Real_t> y8n(numElem8) ;
   std::vector<Real_t> z8n(numElem8) ;

   // Extract raw pointers for lambda capture
   Real_t* dvdx_ptr = dvdx.data() ;
   Real_t* dvdy_ptr = dvdy.data() ;
   Real_t* dvdz_ptr = dvdz.data() ;
   Real_t* x8n_ptr  = x8n.data() ;
   Real_t* y8n_ptr  = y8n.data() ;
   Real_t* z8n_ptr  = z8n.data() ;

   /* start loop over elements */
   Kokkos::parallel_for("CalcHourglassControlForElems", numElem,
                        [&](Index_t i) {
      Real_t  x1[8],  y1[8],  z1[8] ;
      Real_t pfx[8], pfy[8], pfz[8] ;

      Index_t* elemToNode = domain.nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(Index_t ii=0;ii<8;++ii){
         Index_t jj=8*i+ii;

         dvdx_ptr[jj] = pfx[ii];
         dvdy_ptr[jj] = pfy[ii];
         dvdz_ptr[jj] = pfz[ii];
         x8n_ptr[jj]  = x1[ii];
         y8n_ptr[jj]  = y1[ii];
         z8n_ptr[jj]  = z1[ii];
      }

      determ[i] = domain.volo(i) * domain.v(i);

      /* Do a check for negative volumes */
      if ( domain.v(i) <= Real_t(0.0) ) {
         exit(VolumeError);
      }
   });

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems( domain,
                                    determ, x8n_ptr, y8n_ptr, z8n_ptr,
                                    dvdx_ptr, dvdy_ptr, dvdz_ptr,
                                    hgcoef, numElem, domain.numNode()) ;
   }
}

/******************************************/

void CalcVolumeForceForElems(Domain& domain)
{
   Index_t numElem = domain.numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = domain.hgcoef() ;
      std::vector<Real_t> sigxx(numElem) ;
      std::vector<Real_t> sigyy(numElem) ;
      std::vector<Real_t> sigzz(numElem) ;
      std::vector<Real_t> determ(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(domain, sigxx.data(), sigyy.data(), sigzz.data(), numElem);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( domain,
                               sigxx.data(), sigyy.data(), sigzz.data(), determ.data(),
                               numElem, domain.numNode()) ;

      // check for negative element volume
      Real_t* determ_ptr = determ.data() ;
      Kokkos::parallel_for("CalcVolumeForceForElems_check", numElem,
                           [&](Index_t k) {
         if (determ_ptr[k] <= Real_t(0.0)) {
            exit(VolumeError);
         }
      });

      CalcHourglassControlForElems(domain, determ.data(), hgcoef) ;
   }
}
