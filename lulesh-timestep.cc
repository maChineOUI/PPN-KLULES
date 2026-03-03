#include "lulesh-timestep.h"

#if _OPENMP
# include <omp.h>
#endif

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

#if USE_MPI
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif

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
#if _OPENMP
   Index_t threads = omp_get_max_threads();
   static Index_t *courant_elem_per_thread;
   static Real_t *dtcourant_per_thread;
   static bool first = true;
   if (first) {
     courant_elem_per_thread = new Index_t[threads];
     dtcourant_per_thread = new Real_t[threads];
     first = false;
   }
#else
   Index_t threads = 1;
   Index_t courant_elem_per_thread[1];
   Real_t  dtcourant_per_thread[1];
#endif


#pragma omp parallel firstprivate(length, qqc)
   {
      Real_t   qqc2 = Real_t(64.0) * qqc * qqc ;
      Real_t   dtcourant_tmp = dtcourant;
      Index_t  courant_elem  = -1 ;

#if _OPENMP
      Index_t thread_num = omp_get_thread_num();
#else
      Index_t thread_num = 0;
#endif

#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
         Index_t indx = regElemlist[i] ;
         Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

         if ( domain.vdov(indx) < Real_t(0.) ) {
            dtf = dtf
                + qqc2 * domain.arealg(indx) * domain.arealg(indx)
                * domain.vdov(indx) * domain.vdov(indx) ;
         }

         dtf = SQRT(dtf) ;
         dtf = domain.arealg(indx) / dtf ;

         if (domain.vdov(indx) != Real_t(0.)) {
            if ( dtf < dtcourant_tmp ) {
               dtcourant_tmp = dtf ;
               courant_elem  = indx ;
            }
         }
      }

      dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
      courant_elem_per_thread[thread_num] = courant_elem ;
   }

   for (Index_t i = 1; i < threads; ++i) {
      if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
         dtcourant_per_thread[0]    = dtcourant_per_thread[i];
         courant_elem_per_thread[0] = courant_elem_per_thread[i];
      }
   }

   if (courant_elem_per_thread[0] != -1) {
      dtcourant = dtcourant_per_thread[0] ;
   }

   return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
                                 Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
#if _OPENMP
   Index_t threads = omp_get_max_threads();
   static Index_t *hydro_elem_per_thread;
   static Real_t *dthydro_per_thread;
   static bool first = true;
   if (first) {
     hydro_elem_per_thread = new Index_t[threads];
     dthydro_per_thread = new Real_t[threads];
     first = false;
   }
#else
   Index_t threads = 1;
   Index_t hydro_elem_per_thread[1];
   Real_t  dthydro_per_thread[1];
#endif

#pragma omp parallel firstprivate(length, dvovmax)
   {
      Real_t dthydro_tmp = dthydro ;
      Index_t hydro_elem = -1 ;

#if _OPENMP
      Index_t thread_num = omp_get_thread_num();
#else
      Index_t thread_num = 0;
#endif

#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
         Index_t indx = regElemlist[i] ;

         if (domain.vdov(indx) != Real_t(0.)) {
            Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx))+Real_t(1.e-20)) ;

            if ( dthydro_tmp > dtdvov ) {
                  dthydro_tmp = dtdvov ;
                  hydro_elem = indx ;
            }
         }
      }

      dthydro_per_thread[thread_num]    = dthydro_tmp ;
      hydro_elem_per_thread[thread_num] = hydro_elem ;
   }

   for (Index_t i = 1; i < threads; ++i) {
      if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
         dthydro_per_thread[0]    = dthydro_per_thread[i];
         hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
      }
   }

   if (hydro_elem_per_thread[0] != -1) {
      dthydro =  dthydro_per_thread[0] ;
   }

   return ;
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
