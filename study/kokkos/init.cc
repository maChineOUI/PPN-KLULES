#include <math.h>
#if USE_MPI
# include <mpi.h>
#endif
#if _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include "lulesh.h"
#include <Kokkos_Core.hpp>

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost)
   :
   m_e_cut(Real_t(1.0e-7)),
   m_p_cut(Real_t(1.0e-7)),
   m_q_cut(Real_t(1.0e-7)),
   m_v_cut(Real_t(1.0e-10)),
   m_u_cut(Real_t(1.0e-7)),
   m_hgcoef(Real_t(3.0)),
   m_ss4o3(Real_t(4.0)/Real_t(3.0)),
   m_qstop(Real_t(1.0e+12)),
   m_monoq_max_slope(Real_t(1.0)),
   m_monoq_limiter_mult(Real_t(2.0)),
   m_qlc_monoq(Real_t(0.5)),
   m_qqc_monoq(Real_t(2.0)/Real_t(3.0)),
   m_qqc(Real_t(2.0)),
   m_eosvmax(Real_t(1.0e+9)),
   m_eosvmin(Real_t(1.0e-9)),
   m_pmin(Real_t(0.)),
   m_emin(Real_t(-1.0e+15)),
   m_dvovmax(Real_t(0.1)),
   m_refdens(Real_t(1.0))
{
   Index_t edgeElems = nx ;
   Index_t edgeNodes = edgeElems+1 ;
   this->cost() = cost;

   m_tp       = tp ;
   m_numRanks = numRanks ;

   m_colLoc   =   colLoc ;
   m_rowLoc   =   rowLoc ;
   m_planeLoc = planeLoc ;
   
   m_sizeX = edgeElems ;
   m_sizeY = edgeElems ;
   m_sizeZ = edgeElems ;
   m_numElem = edgeElems*edgeElems*edgeElems ;

   m_numNode = edgeNodes*edgeNodes*edgeNodes ;

   // allocate region list view (member is Kokkos::View in the Kokkos header)
   m_regNumList = decltype(m_regNumList)("m_regNumList", numElem());

   // allocate element/node persistent storage (uses Domain::Allocate* which
   // in the header creates Kokkos::View objects)
   AllocateElemPersistent(numElem()) ;
   AllocateNodePersistent(numNode()) ;

   SetupCommBuffers(edgeNodes);

   // initialize element-centered fields using host mirrors then deep_copy

   {
     auto e_h  = Kokkos::create_mirror_view(m_e);
     auto p_h  = Kokkos::create_mirror_view(m_p);
     auto q_h  = Kokkos::create_mirror_view(m_q);
     auto ss_h = Kokkos::create_mirror_view(m_ss);

     for (Index_t i=0; i<numElem(); ++i) {
       e_h(i) = Real_t(0.0);
       p_h(i) = Real_t(0.0);
       q_h(i) = Real_t(0.0);
       ss_h(i)= Real_t(0.0);
     }

     Kokkos::deep_copy(m_e, e_h);
     Kokkos::deep_copy(m_p, p_h);
     Kokkos::deep_copy(m_q, q_h);
     Kokkos::deep_copy(m_ss, ss_h);
   }

   {
     auto v_h = Kokkos::create_mirror_view(m_v);
     for (Index_t i=0; i<numElem(); ++i) v_h(i) = Real_t(1.0);
     Kokkos::deep_copy(m_v, v_h);
   }

   {
     auto xd_h = Kokkos::create_mirror_view(m_xd);
     auto yd_h = Kokkos::create_mirror_view(m_yd);
     auto zd_h = Kokkos::create_mirror_view(m_zd);
     for (Index_t i=0; i<numNode(); ++i) {
       xd_h(i) = Real_t(0.0);
       yd_h(i) = Real_t(0.0);
       zd_h(i) = Real_t(0.0);
     }
     Kokkos::deep_copy(m_xd, xd_h);
     Kokkos::deep_copy(m_yd, yd_h);
     Kokkos::deep_copy(m_zd, zd_h);
   }

   {
     auto xdd_h = Kokkos::create_mirror_view(m_xdd);
     auto ydd_h = Kokkos::create_mirror_view(m_ydd);
     auto zdd_h = Kokkos::create_mirror_view(m_zdd);
     for (Index_t i=0; i<numNode(); ++i) {
       xdd_h(i) = Real_t(0.0);
       ydd_h(i) = Real_t(0.0);
       zdd_h(i) = Real_t(0.0);
     }
     Kokkos::deep_copy(m_xdd, xdd_h);
     Kokkos::deep_copy(m_ydd, ydd_h);
     Kokkos::deep_copy(m_zdd, zdd_h);
   }

   {
     auto nm_h = Kokkos::create_mirror_view(m_nodalMass);
     for (Index_t i=0; i<numNode(); ++i) nm_h(i) = Real_t(0.0);
     Kokkos::deep_copy(m_nodalMass, nm_h);
   }

   // Build mesh (we will fill coordinates and nodelist via host mirrors)
   BuildMesh(nx, edgeNodes, edgeElems);

#if _OPENMP
   SetupThreadSupportStructures();
#endif

   CreateRegionIndexSets(nr, balance);
   SetupSymmetryPlanes(edgeNodes);
   SetupElementConnectivities(edgeElems);
   SetupBoundaryConditions(edgeElems);

   dtfixed() = Real_t(-1.0e-6) ;
   stoptime()  = Real_t(1.0e-2);

   deltatimemultlb() = Real_t(1.1) ;
   deltatimemultub() = Real_t(1.2) ;
   dtcourant() = Real_t(1.0e+20) ;
   dthydro()   = Real_t(1.0e+20) ;
   dtmax()     = Real_t(1.0e-2) ;
   time()    = Real_t(0.) ;
   cycle()   = Int_t(0) ;

   // compute volo, elemMass and nodalMass by using host mirrors for
   // nodelist and coordinate arrays
   {
     auto nodelist_h = Kokkos::create_mirror_view(m_nodelist);
     auto x_h = Kokkos::create_mirror_view(m_x);
     auto y_h = Kokkos::create_mirror_view(m_y);
     auto z_h = Kokkos::create_mirror_view(m_z);

     Kokkos::deep_copy(nodelist_h, m_nodelist);
     Kokkos::deep_copy(x_h, m_x);
     Kokkos::deep_copy(y_h, m_y);
     Kokkos::deep_copy(z_h, m_z);

     auto volo_h = Kokkos::create_mirror_view(m_volo);
     auto elemMass_h = Kokkos::create_mirror_view(m_elemMass);
     auto nodalMass_h = Kokkos::create_mirror_view(m_nodalMass);

     for (Index_t i=0; i<numElem(); ++i) {
       Real_t x_local[8], y_local[8], z_local[8] ;
       for (Index_t lnode=0; lnode<8; ++lnode) {
         Index_t gnode = nodelist_h(i*8 + lnode);
         x_local[lnode] = x_h(gnode);
         y_local[lnode] = y_h(gnode);
         z_local[lnode] = z_h(gnode);
       }
       Real_t volume = CalcElemVolume(x_local, y_local, z_local );
       volo_h(i) = volume ;
       elemMass_h(i) = volume ;
       for (Index_t j=0; j<8; ++j) {
         Index_t idx = nodelist_h(i*8 + j) ;
         nodalMass_h(idx) += volume / Real_t(8.0) ;
       }
     }

     Kokkos::deep_copy(m_volo, volo_h);
     Kokkos::deep_copy(m_elemMass, elemMass_h);
     Kokkos::deep_copy(m_nodalMass, nodalMass_h);
   }

   const Real_t ebase = Real_t(3.948746e+7);
   Real_t scale = (nx*m_tp)/Real_t(45.0);
   Real_t einit = ebase*scale*scale*scale;
   if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
      auto e_h = Kokkos::create_mirror_view(m_e);
      Kokkos::deep_copy(e_h, m_e);
      e_h(0) = einit;
      Kokkos::deep_copy(m_e, e_h);
   }
   deltatime() = (Real_t(.5)*cbrt(m_volo(0)))/sqrt(Real_t(2.0)*einit);
} // End constructor


////////////////////////////////////////////////////////////////////////////////
Domain::~Domain()
{
  // Kokkos::View members automatically release memory on destruction.
  // Host-side std::vectors (used for region/nodal index sets) are also cleaned up automatically.
} // End destructor


////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  Index_t meshEdgeElems = m_tp*nx ;

  auto x_h = Kokkos::create_mirror_view(m_x);
  auto y_h = Kokkos::create_mirror_view(m_y);
  auto z_h = Kokkos::create_mirror_view(m_z);
  auto nodelist_h = Kokkos::create_mirror_view(m_nodelist);

  Index_t nidx = 0 ;
  Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
  for (Index_t plane=0; plane<edgeNodes; ++plane) {
    Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
    for (Index_t row=0; row<edgeNodes; ++row) {
      Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
      for (Index_t col=0; col<edgeNodes; ++col) {
        x_h(nidx) = tx ;
        y_h(nidx) = ty ;
        z_h(nidx) = tz ;
        ++nidx ;
        tx = Real_t(1.125)*Real_t(m_colLoc*nx+col+1)/Real_t(meshEdgeElems) ;
      }
      ty = Real_t(1.125)*Real_t(m_rowLoc*nx+row+1)/Real_t(meshEdgeElems) ;
    }
    tz = Real_t(1.125)*Real_t(m_planeLoc*nx+plane+1)/Real_t(meshEdgeElems) ;
  }

  Index_t zidx = 0 ;
  nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
        nodelist_h(zidx*8 + 0) = nidx;
        nodelist_h(zidx*8 + 1) = nidx + 1;
        nodelist_h(zidx*8 + 2) = nidx + edgeNodes + 1;
        nodelist_h(zidx*8 + 3) = nidx + edgeNodes;
        nodelist_h(zidx*8 + 4) = nidx + edgeNodes*edgeNodes;
        nodelist_h(zidx*8 + 5) = nidx + edgeNodes*edgeNodes + 1;
        nodelist_h(zidx*8 + 6) = nidx + edgeNodes*edgeNodes + edgeNodes + 1;
        nodelist_h(zidx*8 + 7) = nidx + edgeNodes*edgeNodes + edgeNodes;
        ++zidx ;
        ++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }

  Kokkos::deep_copy(m_x, x_h);
  Kokkos::deep_copy(m_y, y_h);
  Kokkos::deep_copy(m_z, z_h);
  Kokkos::deep_copy(m_nodelist, nodelist_h);
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupThreadSupportStructures()
{
#if _OPENMP
   Index_t numthreads = omp_get_max_threads();
#else
   Index_t numthreads = 1;
#endif

  if (numthreads > 1) {
    std::vector<Index_t> nodeElemCount(numNode(), 0);

    {
      auto nodelist_h = Kokkos::create_mirror_view(m_nodelist);
      Kokkos::deep_copy(nodelist_h, m_nodelist);

      for (Index_t i=0; i<numElem(); ++i) {
        for (Index_t j=0; j<8; ++j) {
          Index_t n = nodelist_h(i*8 + j);
          ++nodeElemCount[n];
        }
      }
    }

    m_nodeElemStart_host_vector.resize(numNode()+1);
    m_nodeElemStart_host_vector[0] = 0;
    for (Index_t i=1; i<=numNode(); ++i) {
      m_nodeElemStart_host_vector[i] = m_nodeElemStart_host_vector[i-1] + nodeElemCount[i-1];
    }

    m_nodeElemCornerList_host_vector.resize(m_nodeElemStart_host_vector[numNode()]);

    {
      auto nodelist_h = Kokkos::create_mirror_view(m_nodelist);
      Kokkos::deep_copy(nodelist_h, m_nodelist);

      std::vector<Index_t> tmpCount(numNode(), 0);
      for (Index_t i=0; i<numElem(); ++i) {
        for (Index_t j=0; j<8; ++j) {
          Index_t m = nodelist_h(i*8 + j);
          Index_t k = i*8 + j;
          Index_t offset = m_nodeElemStart_host_vector[m] + tmpCount[m];
          m_nodeElemCornerList_host_vector[offset] = k;
          ++tmpCount[m];
        }
      }
    }

    // basic validation
    Index_t clSize = m_nodeElemStart_host_vector[numNode()];
    for (Index_t i=0; i<clSize; ++i) {
      Index_t clv = m_nodeElemCornerList_host_vector[i];
      if ((clv < 0) || (clv > numElem()*8)) {
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
      }
    }

    // copy host vectors into HostView members for later host-access
    m_nodeElemStart_host_view = HostView1D<Index_t>("nodeElemStart", numNode()+1);
    m_nodeElemCornerList_host_view = HostView1D<Index_t>("nodeElemCornerList", clSize);
    for (Index_t i=0;i<=numNode(); ++i) m_nodeElemStart_host_view(i) = m_nodeElemStart_host_vector[i];
    for (Index_t i=0;i<clSize; ++i) m_nodeElemCornerList_host_view(i) = m_nodeElemCornerList_host_vector[i];
  }
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
  Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ()))+1 ;
  m_maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  m_maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

#if USE_MPI
  Index_t comBufSize =
    (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    m_maxEdgeSize * MAX_FIELDS_PER_MPI_COMM ;

  comBufSize += ((m_rowMin & m_colMin & m_planeMin) +
		 (m_rowMin & m_colMin & m_planeMax) +
		 (m_rowMin & m_colMax & m_planeMin) +
		 (m_rowMin & m_colMax & m_planeMax) +
		 (m_rowMax & m_colMin & m_planeMin) +
		 (m_rowMax & m_colMin & m_planeMax) +
		 (m_rowMax & m_colMax & m_planeMin) +
		 (m_rowMax & m_colMax & m_planeMax)) * CACHE_COHERENCE_PAD_REAL ;

  commDataSend = HostView1D<Real_t>("commSend", comBufSize);
  commDataRecv = HostView1D<Real_t>("commRecv", comBufSize);
  memset(commDataSend.data(), 0, comBufSize*sizeof(Real_t));
  memset(commDataRecv.data(), 0, comBufSize*sizeof(Real_t));
#endif

  if (m_colLoc == 0) m_symmX_host.resize(edgeNodes*edgeNodes);
  if (m_rowLoc == 0) m_symmY_host.resize(edgeNodes*edgeNodes);
  if (m_planeLoc == 0) m_symmZ_host.resize(edgeNodes*edgeNodes);
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
   int myRank = 0;
#endif

   this->numReg() = nr;

   m_regElemSize = decltype(m_regElemSize)("regElemSize", numReg());
   m_regNumList  = decltype(m_regNumList)("regNumList", numElem());

   // fill regNumList on host (we'll keep a host-side vector for region lists)
   auto regNumList_h = Kokkos::create_mirror_view(m_regNumList);
   for (Index_t i=0; i<numElem(); ++i) regNumList_h(i) = 1;
   Kokkos::deep_copy(m_regNumList, regNumList_h);

   // if only 1 region, set sizes accordingly
   if (numReg() == 1) {
     // set regElemSize to zeros, then we'll compute below
     auto regElemSize_h = Kokkos::create_mirror_view(m_regElemSize);
     for (Index_t i=0;i<numReg();++i) regElemSize_h(i)=0;
     Kokkos::deep_copy(m_regElemSize, regElemSize_h);
   } else {
     // For simplicity keep a host-side algorithm similar to original that fills regNumList_host vector.
     std::vector<Index_t> regNumList_host(numElem());
#if USE_MPI
     srand(myRank);
#else
     srand(0);
#endif
     if (numReg() == 1) {
       for (Index_t i=0;i<numElem();++i) regNumList_host[i]=1;
     } else {
       int costDenominator = 0;
       std::vector<int> regBinEnd(numReg());
       for (Index_t i=0;i<numReg();++i) {
         regBinEnd[i] = (i==0 ? (i+1) : regBinEnd[i-1] ) + int(pow((i+1), balance));
         costDenominator = regBinEnd[i];
       }
       Index_t nextIndex = 0;
       Int_t lastReg = -1;
       while (nextIndex < numElem()) {
         Int_t regionVar = rand() % costDenominator;
         Index_t i = 0;
         while (regionVar >= regBinEnd[i]) ++i;
         Int_t regionNum = ((i + myRank) % numReg()) + 1;
         while (regionNum == lastReg) {
           regionVar = rand() % costDenominator;
           i = 0;
           while (regionVar >= regBinEnd[i]) ++i;
           regionNum = ((i + myRank) % numReg()) + 1;
         }
         int binSize = rand() % 1000;
         Index_t elements;
         if(binSize < 773) elements = rand() % 15 + 1;
         else if(binSize < 937) elements = rand() % 16 + 16;
         else if(binSize < 970) elements = rand() % 32 + 32;
         else if(binSize < 974) elements = rand() % 64 + 64;
         else if(binSize < 978) elements = rand() % 128 + 128;
         else if(binSize < 981) elements = rand() % 256 + 256;
         else elements = rand() % 1537 + 512;
         Index_t runto = elements + nextIndex;
         while (nextIndex < runto && nextIndex < numElem()) {
           regNumList_host[nextIndex] = regionNum;
           ++nextIndex;
         }
         lastReg = regionNum;
       }
     }
     // copy to device view
     for (Index_t i=0;i<numElem();++i) regNumList_h(i) = regNumList_host[i];
     Kokkos::deep_copy(m_regNumList, regNumList_h);
   }

   // compute regElemSize on host and fill regElemlist_host (keep host vectors for reg lists)
   m_regElemlist_host.assign(numReg(), std::vector<Index_t>());
   auto regNumList_copy = Kokkos::create_mirror_view(m_regNumList);
   Kokkos::deep_copy(regNumList_copy, m_regNumList);
   for (Index_t i=0;i<numElem();++i) {
     int r = regNumList_copy(i)-1;
     m_regElemlist_host[r].push_back(i);
   }
   // fill device regElemSize
   auto regElemSize_h = Kokkos::create_mirror_view(m_regElemSize);
   for (Index_t r=0;r<numReg();++r) regElemSize_h(r) = static_cast<Index_t>(m_regElemlist_host[r].size());
   Kokkos::deep_copy(m_regElemSize, regElemSize_h);
}

/////////////////////////////////////////////////////////////
void 
Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
  auto symmX_h = &m_symmX_host;
  auto symmY_h = &m_symmY_host;
  auto symmZ_h = &m_symmZ_host;

  Index_t nidx = 0 ;
  for (Index_t i=0; i<edgeNodes; ++i) {
    Index_t planeInc = i*edgeNodes*edgeNodes ;
    Index_t rowInc   = i*edgeNodes ;
    for (Index_t j=0; j<edgeNodes; ++j) {
      if (m_planeLoc == 0) {
        if ((Index_t)symmZ_h->size() > (Index_t)nidx) (*symmZ_h)[nidx] = rowInc + j;
      }
      if (m_rowLoc == 0) {
        if ((Index_t)symmY_h->size() > (Index_t)nidx) (*symmY_h)[nidx] = planeInc + j;
      }
      if (m_colLoc == 0) {
        if ((Index_t)symmX_h->size() > (Index_t)nidx) (*symmX_h)[nidx] = planeInc + j*edgeNodes;
      }
      ++nidx ;
    }
  }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Int_t edgeElems)
{
   // accessors were KOKKOS_INLINE; here we use host mirrors to set device views
   auto lxim_h  = Kokkos::create_mirror_view(m_lxim);
   auto lxip_h  = Kokkos::create_mirror_view(m_lxip);
   auto letam_h = Kokkos::create_mirror_view(m_letam);
   auto letap_h = Kokkos::create_mirror_view(m_letap);
   auto lzetam_h= Kokkos::create_mirror_view(m_lzetam);
   auto lzetap_h= Kokkos::create_mirror_view(m_lzetap);

   for (Index_t i=0;i<numElem();++i) {
     lxim_h(i)=0; lxip_h(i)=0; letam_h(i)=0; letap_h(i)=0; lzetam_h(i)=0; lzetap_h(i)=0;
   }

   lxim_h(0) = 0 ;
   for (Index_t i=1; i<numElem(); ++i) {
      lxim_h(i)   = i-1 ;
      lxip_h(i-1) = i ;
   }
   lxip_h(numElem()-1) = numElem()-1 ;

   for (Index_t i=0; i<edgeElems; ++i) {
      letam_h(i) = i ; 
      letap_h(numElem()-edgeElems+i) = numElem()-edgeElems+i ;
   }
   for (Index_t i=edgeElems; i<numElem(); ++i) {
      letam_h(i) = i-edgeElems ;
      letap_h(i-edgeElems) = i ;
   }

   for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
      lzetam_h(i) = i ;
      lzetap_h(numElem()-edgeElems*edgeElems+i) = numElem()-edgeElems*edgeElems+i ;
   }
   for (Index_t i=edgeElems*edgeElems; i<numElem(); ++i) {
      lzetam_h(i) = i - edgeElems*edgeElems ;
      lzetap_h(i-edgeElems*edgeElems) = i ;
   }

   Kokkos::deep_copy(m_lxim, lxim_h);
   Kokkos::deep_copy(m_lxip, lxip_h);
   Kokkos::deep_copy(m_letam, letam_h);
   Kokkos::deep_copy(m_letap, letap_h);
   Kokkos::deep_copy(m_lzetam, lzetam_h);
   Kokkos::deep_copy(m_lzetap, lzetap_h);
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Int_t edgeElems) 
{
  auto elemBC_h = Kokkos::create_mirror_view(m_elemBC);
  for (Index_t i=0;i<numElem();++i) elemBC_h(i) = Int_t(0);
  Kokkos::deep_copy(m_elemBC, elemBC_h);

  Index_t ghostIdx[6];
  for (int ii=0; ii<6; ++ii) ghostIdx[ii] = INT_MIN;

  Int_t pidx = numElem();
  if (m_planeMin != 0) {
    ghostIdx[0] = pidx;
    pidx += sizeX()*sizeY();
  }
  if (m_planeMax != 0) {
    ghostIdx[1] = pidx;
    pidx += sizeX()*sizeY();
  }
  if (m_rowMin != 0) {
    ghostIdx[2] = pidx;
    pidx += sizeX()*sizeZ();
  }
  if (m_rowMax != 0) {
    ghostIdx[3] = pidx;
    pidx += sizeX()*sizeZ();
  }
  if (m_colMin != 0) {
    ghostIdx[4] = pidx;
    pidx += sizeY()*sizeZ();
  }
  if (m_colMax != 0) {
    ghostIdx[5] = pidx;
  }

  auto lxim_h  = Kokkos::create_mirror_view(m_lxim);
  auto lxip_h  = Kokkos::create_mirror_view(m_lxip);
  auto letam_h = Kokkos::create_mirror_view(m_letam);
  auto letap_h = Kokkos::create_mirror_view(m_letap);
  auto lzetam_h= Kokkos::create_mirror_view(m_lzetam);
  auto lzetap_h= Kokkos::create_mirror_view(m_lzetap);

  auto elemBC_copy = Kokkos::create_mirror_view(m_elemBC);
  Kokkos::deep_copy(elemBC_copy, m_elemBC);

  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (m_planeLoc == 0) {
        elemBC_copy(rowInc+j) |= ZETA_M_SYMM ;
      }
      else {
        elemBC_copy(rowInc+j) |= ZETA_M_COMM ;
        lzetam_h(rowInc+j) = ghostIdx[0] + rowInc + j ;
      }

      if (m_planeLoc == m_tp-1) {
        elemBC_copy(rowInc+j+numElem()-edgeElems*edgeElems) |= ZETA_P_FREE;
      }
      else {
        elemBC_copy(rowInc+j+numElem()-edgeElems*edgeElems) |= ZETA_P_COMM ;
        lzetap_h(rowInc+j+numElem()-edgeElems*edgeElems) = ghostIdx[1] + rowInc + j ;
      }

      if (m_rowLoc == 0) {
        elemBC_copy(planeInc+j) |= ETA_M_SYMM ;
      }
      else {
        elemBC_copy(planeInc+j) |= ETA_M_COMM ;
        letam_h(planeInc+j) = ghostIdx[2] + rowInc + j ;
      }

      if (m_rowLoc == m_tp-1) {
        elemBC_copy(planeInc+j+edgeElems*edgeElems-edgeElems) |= ETA_P_FREE ;
      }
      else {
        elemBC_copy(planeInc+j+edgeElems*edgeElems-edgeElems) |= ETA_P_COMM ;
        letap_h(planeInc+j+edgeElems*edgeElems-edgeElems) = ghostIdx[3] +  rowInc + j ;
      }

      if (m_colLoc == 0) {
        elemBC_copy(planeInc+j*edgeElems) |= XI_M_SYMM ;
      }
      else {
        elemBC_copy(planeInc+j*edgeElems) |= XI_M_COMM ;
        lxim_h(planeInc+j*edgeElems) = ghostIdx[4] + rowInc + j ;
      }

      if (m_colLoc == m_tp-1) {
        elemBC_copy(planeInc+j*edgeElems+edgeElems-1) |= XI_P_FREE ;
      }
      else {
        elemBC_copy(planeInc+j*edgeElems+edgeElems-1) |= XI_P_COMM ;
        lxip_h(planeInc+j*edgeElems+edgeElems-1) = ghostIdx[5] + rowInc + j ;
      }
    }
  }

  Kokkos::deep_copy(m_elemBC, elemBC_copy);
  Kokkos::deep_copy(m_lxim, lxim_h);
  Kokkos::deep_copy(m_lxip, lxip_h);
  Kokkos::deep_copy(m_letam, letam_h);
  Kokkos::deep_copy(m_letap, letap_h);
  Kokkos::deep_copy(m_lzetam, lzetam_h);
  Kokkos::deep_copy(m_lzetap, lzetap_h);
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;
   
   testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("MPI operations only support float and double right now...\n");
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   Int_t remainder = dx*dy*dz % numRanks ;
   if (myRank < remainder) {
      myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
   }
   else {
      myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
         (myRank - remainder)*(dx*dy*dz/numRanks) ;
   }

   *col = myDom % dx ;
   *row = (myDom / dx) % dy ;
   *plane = myDom / (dx*dy) ;
   *side = testProcs;

   return;
}

