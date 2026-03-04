#include "lulesh-eos.h"

/******************************************/

static inline
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length, Index_t *regElemList)
{
   Kokkos::parallel_for("CalcPressureForElems_bvc", length,
                        [&](Index_t i) {
      Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
      bvc[i] = c1s * (compression[i] + Real_t(1.));
      pbvc[i] = c1s;
   });

   Kokkos::parallel_for("CalcPressureForElems_p", length,
                        [&](Index_t i) {
      Index_t elem = regElemList[i];

      p_new[i] = bvc[i] * e_old[i] ;

      if    (std::fabs(p_new[i]) <  p_cut   )
         p_new[i] = Real_t(0.0) ;

      if    ( vnewc[elem] >= eosvmax ) /* impossible condition here? */
         p_new[i] = Real_t(0.0) ;

      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
   });
}

/******************************************/

static inline
void CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old, Real_t* e_old, Real_t* q_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
                        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                        Real_t* qq_old, Real_t* ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length, Index_t *regElemList)
{
   std::vector<Real_t> pHalfStep(length) ;
   Real_t* pHalfStep_ptr = pHalfStep.data() ;

   Kokkos::parallel_for("CalcEnergyForElems_e1", length,
                        [&](Index_t i) {
      e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
         + Real_t(0.5) * work[i];

      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   });

   CalcPressureForElems(pHalfStep_ptr, bvc, pbvc, e_new, compHalfStep, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

   Kokkos::parallel_for("CalcEnergyForElems_q1", length,
                        [&](Index_t i) {
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

      if ( delvc[i] > Real_t(0.) ) {
         q_new[i] /* = qq_old[i] = ql_old[i] */ = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep_ptr[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = std::sqrt(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new[i] + Real_t(0.5) * delvc[i]
         * (  Real_t(3.0)*(p_old[i]     + q_old[i])
              - Real_t(4.0)*(pHalfStep_ptr[i] + q_new[i])) ;
   });

   Kokkos::parallel_for("CalcEnergyForElems_e2", length,
                        [&](Index_t i) {
      e_new[i] += Real_t(0.5) * work[i];

      if (std::fabs(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   });

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

   Kokkos::parallel_for("CalcEnergyForElems_e3", length,
                        [&](Index_t i) {
      const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
      Index_t elem = regElemList[i];
      Real_t q_tilde ;

      if (delvc[i] > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = std::sqrt(ssc) ;
         }

         q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
                               - Real_t(8.0)*(pHalfStep_ptr[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (std::fabs(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   });

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

   Kokkos::parallel_for("CalcEnergyForElems_q2", length,
                        [&](Index_t i) {
      Index_t elem = regElemList[i];

      if ( delvc[i] <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = std::sqrt(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;

         if (std::fabs(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
      }
   });

   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(Domain &domain,
                            Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3,
                            Index_t len, Index_t *regElemList)
{
   Kokkos::parallel_for("CalcSoundSpeedForElems", len,
                        [&](Index_t i) {
      Index_t elem = regElemList[i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[elem] * vnewc[elem] *
                 bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= Real_t(.1111111e-36)) {
         ssTmp = Real_t(.3333333e-18);
      }
      else {
         ssTmp = std::sqrt(ssTmp);
      }
      domain.ss(elem) = ssTmp ;
   });
}

/******************************************/

static inline
void EvalEOSForElems(Domain& domain, Real_t *vnewc,
                     Int_t numElemReg, Index_t *regElemList, Int_t rep)
{
   Real_t  e_cut = domain.e_cut() ;
   Real_t  p_cut = domain.p_cut() ;
   Real_t  ss4o3 = domain.ss4o3() ;
   Real_t  q_cut = domain.q_cut() ;

   Real_t eosvmax = domain.eosvmax() ;
   Real_t eosvmin = domain.eosvmin() ;
   Real_t pmin    = domain.pmin() ;
   Real_t emin    = domain.emin() ;
   Real_t rho0    = domain.refdens() ;

   // These temporaries will be of different size for
   // each call (due to different sized region element
   // lists)
   std::vector<Real_t> e_old(numElemReg) ;
   std::vector<Real_t> delvc(numElemReg) ;
   std::vector<Real_t> p_old(numElemReg) ;
   std::vector<Real_t> q_old(numElemReg) ;
   std::vector<Real_t> compression(numElemReg) ;
   std::vector<Real_t> compHalfStep(numElemReg) ;
   std::vector<Real_t> qq_old(numElemReg) ;
   std::vector<Real_t> ql_old(numElemReg) ;
   std::vector<Real_t> work(numElemReg) ;
   std::vector<Real_t> p_new(numElemReg) ;
   std::vector<Real_t> e_new(numElemReg) ;
   std::vector<Real_t> q_new(numElemReg) ;
   std::vector<Real_t> bvc(numElemReg) ;
   std::vector<Real_t> pbvc(numElemReg) ;

   // Extract raw pointers for lambda capture
   Real_t* e_old_ptr        = e_old.data() ;
   Real_t* delvc_ptr        = delvc.data() ;
   Real_t* p_old_ptr        = p_old.data() ;
   Real_t* q_old_ptr        = q_old.data() ;
   Real_t* compression_ptr  = compression.data() ;
   Real_t* compHalfStep_ptr = compHalfStep.data() ;
   Real_t* qq_old_ptr       = qq_old.data() ;
   Real_t* ql_old_ptr       = ql_old.data() ;
   Real_t* work_ptr         = work.data() ;
   Real_t* p_new_ptr        = p_new.data() ;
   Real_t* e_new_ptr        = e_new.data() ;
   Real_t* q_new_ptr        = q_new.data() ;
   Real_t* bvc_ptr          = bvc.data() ;
   Real_t* pbvc_ptr         = pbvc.data() ;

   //loop to add load imbalance based on region number
   for(Int_t j = 0; j < rep; j++) {
      /* compress data, minimal set */
      Kokkos::parallel_for("EvalEOSForElems_gather", numElemReg,
                           [&](Index_t i) {
         Index_t elem = regElemList[i];
         e_old_ptr[i]  = domain.e(elem) ;
         delvc_ptr[i]  = domain.delv(elem) ;
         p_old_ptr[i]  = domain.p(elem) ;
         q_old_ptr[i]  = domain.q(elem) ;
         qq_old_ptr[i] = domain.qq(elem) ;
         ql_old_ptr[i] = domain.ql(elem) ;
      });

      Kokkos::parallel_for("EvalEOSForElems_compression", numElemReg,
                           [&](Index_t i) {
         Index_t elem = regElemList[i];
         Real_t vchalf ;
         compression_ptr[i] = Real_t(1.) / vnewc[elem] - Real_t(1.);
         vchalf = vnewc[elem] - delvc_ptr[i] * Real_t(.5);
         compHalfStep_ptr[i] = Real_t(1.) / vchalf - Real_t(1.);
      });

      /* Check for v > eosvmax or v < eosvmin */
      if ( eosvmin != Real_t(0.) ) {
         Kokkos::parallel_for("EvalEOSForElems_eosvmin", numElemReg,
                              [&](Index_t i) {
            Index_t elem = regElemList[i];
            if (vnewc[elem] <= eosvmin) { /* impossible due to calling func? */
               compHalfStep_ptr[i] = compression_ptr[i] ;
            }
         });
      }
      if ( eosvmax != Real_t(0.) ) {
         Kokkos::parallel_for("EvalEOSForElems_eosvmax", numElemReg,
                              [&](Index_t i) {
            Index_t elem = regElemList[i];
            if (vnewc[elem] >= eosvmax) { /* impossible due to calling func? */
               p_old_ptr[i]        = Real_t(0.) ;
               compression_ptr[i]  = Real_t(0.) ;
               compHalfStep_ptr[i] = Real_t(0.) ;
            }
         });
      }

      Kokkos::parallel_for("EvalEOSForElems_work", numElemReg,
                           [&](Index_t i) {
         work_ptr[i] = Real_t(0.) ;
      });

      CalcEnergyForElems(p_new_ptr, e_new_ptr, q_new_ptr, bvc_ptr, pbvc_ptr,
                         p_old_ptr, e_old_ptr, q_old_ptr, compression_ptr, compHalfStep_ptr,
                         vnewc, work_ptr, delvc_ptr, pmin,
                         p_cut, e_cut, q_cut, emin,
                         qq_old_ptr, ql_old_ptr, rho0, eosvmax,
                         numElemReg, regElemList);
   }

   Kokkos::parallel_for("EvalEOSForElems_scatter", numElemReg,
                        [&](Index_t i) {
      Index_t elem = regElemList[i];
      domain.p(elem) = p_new_ptr[i] ;
      domain.e(elem) = e_new_ptr[i] ;
      domain.q(elem) = q_new_ptr[i] ;
   });

   CalcSoundSpeedForElems(domain,
                          vnewc, rho0, e_new_ptr, p_new_ptr,
                          pbvc_ptr, bvc_ptr, ss4o3,
                          numElemReg, regElemList) ;
}

/******************************************/

void ApplyMaterialPropertiesForElems(Domain& domain, Real_t vnew[])
{
   Index_t numElem = domain.numElem() ;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = domain.eosvmin() ;
    Real_t eosvmax = domain.eosvmax() ;

    // Bound the updated relative volumes with eosvmin/max
    if (eosvmin != Real_t(0.)) {
       Kokkos::parallel_for("ApplyMaterialProperties_eosvmin", numElem,
                            [&](Index_t i) {
          if (vnew[i] < eosvmin)
             vnew[i] = eosvmin ;
       });
    }

    if (eosvmax != Real_t(0.)) {
       Kokkos::parallel_for("ApplyMaterialProperties_eosvmax", numElem,
                            [&](Index_t i) {
          if (vnew[i] > eosvmax)
             vnew[i] = eosvmax ;
       });
    }

    // This check may not make perfect sense in LULESH, but
    // it's representative of something in the full code -
    // just leave it in, please
    Kokkos::parallel_for("ApplyMaterialProperties_check", numElem,
                         [&](Index_t i) {
       Real_t vc = domain.v(i) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = eosvmin ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = eosvmax ;
       }
       if (vc <= 0.) {
          exit(VolumeError);
       }
    });

    for (Int_t r=0 ; r<domain.numReg() ; r++) {
       Index_t numElemReg = domain.regElemSize(r);
       Index_t *regElemList = domain.regElemlist(r);
       Int_t rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       if(r < domain.numReg()/2)
	 rep = 1;
       //you don't get an expensive region unless you at least have 5 regions
       else if(r < (domain.numReg() - (domain.numReg()+15)/20))
         rep = 1 + domain.cost();
       //very expensive regions
       else
	 rep = 10 * (1+ domain.cost());
       EvalEOSForElems(domain, vnew, numElemReg, regElemList, rep);
    }

  }
}

/******************************************/

void UpdateVolumesForElems(Domain &domain, Real_t *vnew,
                           Real_t v_cut, Index_t length)
{
   if (length != 0) {
      Kokkos::parallel_for("UpdateVolumesForElems", length,
                           [&](Index_t i) {
         Real_t tmpV = vnew[i] ;

         if ( std::fabs(tmpV - Real_t(1.0)) < v_cut )
            tmpV = Real_t(1.0) ;

         domain.v(i) = tmpV ;
      });
   }

   return ;
}
