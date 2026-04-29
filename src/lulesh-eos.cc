#include "lulesh-eos.h"

/******************************************/

static inline
void CalcEnergyForElems(Kokkos::View<Real_t*> p_new,
                        Kokkos::View<Real_t*> e_new,
                        Kokkos::View<Real_t*> q_new,
                        Kokkos::View<Real_t*> bvc,
                        Kokkos::View<Real_t*> pbvc,
                        Kokkos::View<Real_t*> p_old,
                        Kokkos::View<Real_t*> e_old,
                        Kokkos::View<Real_t*> q_old,
                        Kokkos::View<Real_t*> compression,
                        Kokkos::View<Real_t*> compHalfStep,
                        Kokkos::View<Real_t*> vnewc,
                        Kokkos::View<Real_t*> work,
                        Kokkos::View<Real_t*> delvc,
                        Real_t pmin,
                        Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin,
                        Kokkos::View<Real_t*> qq_old,
                        Kokkos::View<Real_t*> ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length,
                        Kokkos::View<Index_t*> regElemList)
{
   /* Fused kernel: replaces 11 parallel_for with a single per-element pass. */
   const Real_t c1s   = Real_t(2.0) / Real_t(3.0) ;
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;

   Kokkos::parallel_for("CalcEnergyForElems", length,
                        KOKKOS_LAMBDA(Index_t i) {
      Index_t elem = regElemList(i) ;

      // --- e1 ---
      Real_t e_i = e_old(i) - Real_t(0.5) * delvc(i) * (p_old(i) + q_old(i))
                             + Real_t(0.5) * work(i) ;
      if (e_i < emin) e_i = emin ;

      // --- 1st CalcPressureForElems (compHalfStep) ---
      Real_t bvc_half = c1s * (compHalfStep(i) + Real_t(1.)) ;
      Real_t phs      = bvc_half * e_i ;
      if (Kokkos::fabs(phs) < p_cut)  phs = Real_t(0.0) ;
      if (vnewc(elem) >= eosvmax)     phs = Real_t(0.0) ;
      if (phs < pmin)                 phs = pmin ;

      // --- q1 ---
      Real_t q_i ;
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep(i)) ;
      if (delvc(i) > Real_t(0.)) {
         q_i = Real_t(0.) ;
      }
      else {
         Real_t ssc = (c1s * e_i + vhalf * vhalf * bvc_half * phs) / rho0 ;
         if (ssc <= Real_t(.1111111e-36)) ssc = Real_t(.3333333e-18) ;
         else                            ssc = Kokkos::sqrt(ssc) ;
         q_i = ssc * ql_old(i) + qq_old(i) ;
      }
      e_i += Real_t(0.5) * delvc(i)
             * (Real_t(3.0)*(p_old(i) + q_old(i)) - Real_t(4.0)*(phs + q_i)) ;

      // --- e2 ---
      e_i += Real_t(0.5) * work(i) ;
      if (Kokkos::fabs(e_i) < e_cut) e_i = Real_t(0.) ;
      if (e_i < emin)                e_i = emin ;

      // --- 2nd CalcPressureForElems (compression) ---
      Real_t bvc_full    = c1s * (compression(i) + Real_t(1.)) ;
      Real_t p_new_local = bvc_full * e_i ;
      if (Kokkos::fabs(p_new_local) < p_cut) p_new_local = Real_t(0.0) ;
      if (vnewc(elem) >= eosvmax)            p_new_local = Real_t(0.0) ;
      if (p_new_local < pmin)                p_new_local = pmin ;

      // --- e3 ---
      Real_t q_tilde ;
      if (delvc(i) > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = (c1s * e_i
                       + vnewc(elem) * vnewc(elem) * bvc_full * p_new_local) / rho0 ;
         if (ssc <= Real_t(.1111111e-36)) ssc = Real_t(.3333333e-18) ;
         else                            ssc = Kokkos::sqrt(ssc) ;
         q_tilde = ssc * ql_old(i) + qq_old(i) ;
      }
      e_i -= (Real_t(7.0)*(p_old(i) + q_old(i))
              - Real_t(8.0)*(phs + q_i)
              + (p_new_local + q_tilde)) * delvc(i) * sixth ;
      if (Kokkos::fabs(e_i) < e_cut) e_i = Real_t(0.) ;
      if (e_i < emin)                e_i = emin ;

      // --- 3rd CalcPressureForElems ---
      Real_t p_i = bvc_full * e_i ;
      if (Kokkos::fabs(p_i) < p_cut) p_i = Real_t(0.0) ;
      if (vnewc(elem) >= eosvmax)    p_i = Real_t(0.0) ;
      if (p_i < pmin)                p_i = pmin ;

      // Write output arrays (bvc/pbvc needed by CalcSoundSpeedForElems)
      p_new(i) = p_i ;
      e_new(i) = e_i ;
      q_new(i) = q_i ;
      bvc(i)   = bvc_full ;
      pbvc(i)  = c1s ;

      // --- q2 ---
      if (delvc(i) <= Real_t(0.)) {
         Real_t ssc = (c1s * e_i
                       + vnewc(elem) * vnewc(elem) * bvc_full * p_i) / rho0 ;
         if (ssc <= Real_t(.1111111e-36)) ssc = Real_t(.3333333e-18) ;
         else                            ssc = Kokkos::sqrt(ssc) ;
         q_new(i) = ssc * ql_old(i) + qq_old(i) ;
         if (Kokkos::fabs(q_new(i)) < q_cut) q_new(i) = Real_t(0.) ;
      }
   });
}

/******************************************/

static inline
void EvalEOSForElems(Domain& domain, Kokkos::View<Real_t*> vnewc,
                     Int_t numElemReg, Index_t *regElemRaw, Int_t rep)
{
   Real_t  e_cut = domain.e_cut() ;
   Real_t  p_cut = domain.p_cut() ;
   Real_t  q_cut = domain.q_cut() ;

   Real_t eosvmax = domain.eosvmax() ;
   Real_t eosvmin = domain.eosvmin() ;
   Real_t pmin    = domain.pmin() ;
   Real_t emin    = domain.emin() ;
   Real_t rho0    = domain.refdens() ;

   // Reuse pre-allocated EOS temporaries from Domain
   auto& t = domain.eosTemps();

   // CPU backends (OpenMP/Serial): zero-copy — wrap the host array directly,
   // no allocation and no copy.
   // GPU backend (CUDA): regElemRaw lives in HostSpace; allocate device
   // memory and transfer before the kernels run.
#ifdef KOKKOS_ENABLE_CUDA
   Kokkos::View<const Index_t*, Kokkos::HostSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
       regElemList_h(regElemRaw, numElemReg) ;
   Kokkos::View<Index_t*> regElemList("regElemList", numElemReg) ;
   Kokkos::deep_copy(regElemList, regElemList_h) ;
#else
   Kokkos::View<Index_t*, Kokkos::HostSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
       regElemList(regElemRaw, numElemReg) ;
#endif

   // Extract domain Views needed by scatter kernels
   auto e_v    = domain.m_elems.m_e ;
   auto delv_v = domain.m_elems.m_delv ;
   auto p_v    = domain.m_elems.m_p ;
   auto q_v    = domain.m_elems.m_q ;
   auto qq_v   = domain.m_elems.m_qq ;
   auto ql_v   = domain.m_elems.m_ql ;
   auto ss_v   = domain.m_elems.m_ss ;

   // EOS temp Views (captured by value in KOKKOS_LAMBDA)
   auto e_old_v        = t.e_old ;
   auto delvc_v        = t.delvc ;
   auto p_old_v        = t.p_old ;
   auto q_old_v        = t.q_old ;
   auto compression_v  = t.compression ;
   auto compHalfStep_v = t.compHalfStep ;
   auto qq_old_v       = t.qq_old ;
   auto ql_old_v       = t.ql_old ;
   auto work_v         = t.work ;
   auto p_new_v        = t.p_new ;
   auto e_new_v        = t.e_new ;
   auto q_new_v        = t.q_new ;
   auto bvc_v          = t.bvc ;
   auto pbvc_v         = t.pbvc ;

   for (Int_t j = 0; j < rep; j++) {
      /* Fused prep: gather + compression + eosvmin/max clamp + zero work */
      Kokkos::parallel_for("EvalEOSForElems_prep", numElemReg,
                           KOKKOS_LAMBDA(Index_t i) {
         Index_t elem = regElemList(i) ;
         e_old_v(i)  = e_v(elem) ;
         delvc_v(i)  = delv_v(elem) ;
         p_old_v(i)  = p_v(elem) ;
         q_old_v(i)  = q_v(elem) ;
         qq_old_v(i) = qq_v(elem) ;
         ql_old_v(i) = ql_v(elem) ;

         Real_t vchalf ;
         compression_v(i)  = Real_t(1.) / vnewc(elem) - Real_t(1.) ;
         vchalf             = vnewc(elem) - delvc_v(i) * Real_t(.5) ;
         compHalfStep_v(i) = Real_t(1.) / vchalf - Real_t(1.) ;

         if (eosvmin != Real_t(0.) && vnewc(elem) <= eosvmin)
            compHalfStep_v(i) = compression_v(i) ;

         if (eosvmax != Real_t(0.) && vnewc(elem) >= eosvmax) {
            p_old_v(i)        = Real_t(0.) ;
            compression_v(i)  = Real_t(0.) ;
            compHalfStep_v(i) = Real_t(0.) ;
         }

         work_v(i) = Real_t(0.) ;
      });

      CalcEnergyForElems(p_new_v, e_new_v, q_new_v, bvc_v, pbvc_v,
                         p_old_v, e_old_v, q_old_v, compression_v, compHalfStep_v,
                         vnewc, work_v, delvc_v, pmin,
                         p_cut, e_cut, q_cut, emin,
                         qq_old_v, ql_old_v, rho0, eosvmax,
                         numElemReg, regElemList) ;
   }

   /* Fused: scatter p/e/q + compute sound speed */
   Kokkos::parallel_for("EvalEOSForElems_scatter_ss", numElemReg,
                        KOKKOS_LAMBDA(Index_t i) {
      Index_t elem = regElemList(i) ;
      p_v(elem) = p_new_v(i) ;
      e_v(elem) = e_new_v(i) ;
      q_v(elem) = q_new_v(i) ;
      Real_t ssTmp = (pbvc_v(i) * e_new_v(i)
                     + vnewc(elem) * vnewc(elem) * bvc_v(i) * p_new_v(i)) / rho0 ;
      if (ssTmp <= Real_t(.1111111e-36)) ssTmp = Real_t(.3333333e-18) ;
      else                               ssTmp = Kokkos::sqrt(ssTmp) ;
      ss_v(elem) = ssTmp ;
   });
}

/******************************************/

void ApplyMaterialPropertiesForElems(Domain& domain, Kokkos::View<Real_t*> vnew)
{
   Index_t numElem = domain.numElem() ;

  if (numElem != 0) {
    Real_t eosvmin = domain.eosvmin() ;
    Real_t eosvmax = domain.eosvmax() ;

    auto v_v = domain.m_elems.m_v ;

    /* Fused: eosvmin/max clamp + volume sanity check */
    Kokkos::parallel_for("ApplyMaterialProperties", numElem,
                         KOKKOS_LAMBDA(Index_t i) {
       if (eosvmin != Real_t(0.) && vnew(i) < eosvmin) vnew(i) = eosvmin ;
       if (eosvmax != Real_t(0.) && vnew(i) > eosvmax) vnew(i) = eosvmax ;
       Real_t vc = v_v(i) ;
       if (eosvmin != Real_t(0.) && vc < eosvmin) vc = eosvmin ;
       if (eosvmax != Real_t(0.) && vc > eosvmax) vc = eosvmax ;
       if (vc <= Real_t(0.))
          Kokkos::abort("VolumeError: non-positive volume in ApplyMaterialProperties") ;
    });

    for (Int_t r = 0; r < domain.numReg(); r++) {
       Index_t numElemReg  = domain.regElemSize(r) ;
       Index_t *regElemRaw = domain.regElemlist(r) ;
       Int_t rep ;
       if (r < domain.numReg()/2)
          rep = 1 ;
       else if (r < (domain.numReg() - (domain.numReg()+15)/20))
          rep = 1 + domain.cost() ;
       else
          rep = 10 * (1 + domain.cost()) ;
       EvalEOSForElems(domain, vnew, numElemReg, regElemRaw, rep) ;
    }
  }
}

/******************************************/

void UpdateVolumesForElems(Domain& domain, Kokkos::View<Real_t*> vnew,
                           Real_t v_cut, Index_t length)
{
   if (length != 0) {
      auto v_v = domain.m_elems.m_v ;

      Kokkos::parallel_for("UpdateVolumesForElems", length,
                           KOKKOS_LAMBDA(Index_t i) {
         Real_t tmpV = vnew(i) ;
         if (Kokkos::fabs(tmpV - Real_t(1.0)) < v_cut)
            tmpV = Real_t(1.0) ;
         v_v(i) = tmpV ;
      });
   }
}
