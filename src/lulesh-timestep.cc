#include "lulesh-timestep.h"

/******************************************/

void TimeIncrement(Domain& domain)
{
   Real_t targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain.deltatime() ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / Real_t(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

      newdt = gnewdt;

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain.deltatime()) &&
       (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

/******************************************/

void CalcTimeConstraintsForElems(Domain& domain) {

   /* Opt-11: flatten 11 per-region parallel_reduce into one over all numElem.
      All elements are processed identically regardless of region membership.
      Saves (numReg - 1) = 10 barriers per step. */

   const Real_t qqc2    = Real_t(64.0) * domain.qqc() * domain.qqc() ;
   const Real_t dvovmax = domain.dvovmax() ;

   Real_t dtc = Real_t(1.0e+20) ;
   Real_t dth = Real_t(1.0e+20) ;

   Kokkos::parallel_reduce("CalcTimeConstraintsForElems", domain.numElem(),
      [&](Index_t indx, Real_t& lc, Real_t& lh) {
         if (domain.vdov(indx) != Real_t(0.)) {
            // Courant constraint
            Real_t dtf = domain.ss(indx) * domain.ss(indx) ;
            if (domain.vdov(indx) < Real_t(0.))
               dtf += qqc2 * domain.arealg(indx) * domain.arealg(indx)
                           * domain.vdov(indx) * domain.vdov(indx) ;
            dtf = domain.arealg(indx) / std::sqrt(dtf) ;
            if (dtf < lc) lc = dtf ;
            // Hydro constraint
            Real_t dtdvov = dvovmax / (std::fabs(domain.vdov(indx)) + Real_t(1.e-20)) ;
            if (dtdvov < lh) lh = dtdvov ;
         }
      },
      Kokkos::Min<Real_t>(dtc),
      Kokkos::Min<Real_t>(dth)
   ) ;

   domain.dtcourant() = dtc ;
   domain.dthydro()   = dth ;
}
