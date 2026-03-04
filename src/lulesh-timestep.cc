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

static inline
void CalcCourantConstraintForElems(Domain &domain, Index_t length,
                                   Index_t *regElemlist,
                                   Real_t qqc, Real_t& dtcourant)
{
   const Real_t qqc2 = Real_t(64.0) * qqc * qqc ;

   using minloc_t = Kokkos::MinLoc<Real_t, Index_t> ;
   typename minloc_t::value_type result ;

   Kokkos::parallel_reduce("CalcCourantConstraintForElems", length,
      [&](Index_t i, typename minloc_t::value_type& upd) {
         Index_t indx = regElemlist[i] ;

         if (domain.vdov(indx) != Real_t(0.)) {
            Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

            if (domain.vdov(indx) < Real_t(0.)) {
               dtf += qqc2 * domain.arealg(indx) * domain.arealg(indx)
                           * domain.vdov(indx) * domain.vdov(indx) ;
            }

            dtf = std::sqrt(dtf) ;
            dtf = domain.arealg(indx) / dtf ;

            if (dtf < upd.val) {
               upd.val = dtf ;
               upd.loc = indx ;
            }
         }
      },
      minloc_t(result)
   );

   if (result.val < dtcourant) {
      dtcourant = result.val ;
   }
}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
                                 Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
   using minloc_t = Kokkos::MinLoc<Real_t, Index_t> ;
   typename minloc_t::value_type result ;

   Kokkos::parallel_reduce("CalcHydroConstraintForElems", length,
      [&](Index_t i, typename minloc_t::value_type& upd) {
         Index_t indx = regElemlist[i] ;

         if (domain.vdov(indx) != Real_t(0.)) {
            Real_t dtdvov = dvovmax / (std::fabs(domain.vdov(indx)) + Real_t(1.e-20)) ;

            if (dtdvov < upd.val) {
               upd.val = dtdvov ;
               upd.loc = indx ;
            }
         }
      },
      minloc_t(result)
   );

   if (result.val < dthydro) {
      dthydro = result.val ;
   }
}

/******************************************/

void CalcTimeConstraintsForElems(Domain& domain) {

   // Initialize conditions to a very large value
   domain.dtcourant() = 1.0e+20;
   domain.dthydro() = 1.0e+20;

   for (Index_t r=0 ; r < domain.numReg() ; ++r) {
      /* evaluate time constraint */
      CalcCourantConstraintForElems(domain, domain.regElemSize(r),
                                    domain.regElemlist(r),
                                    domain.qqc(),
                                    domain.dtcourant()) ;

      /* check hydro constraint */
      CalcHydroConstraintForElems(domain, domain.regElemSize(r),
                                  domain.regElemlist(r),
                                  domain.dvovmax(),
                                  domain.dthydro()) ;
   }
}
