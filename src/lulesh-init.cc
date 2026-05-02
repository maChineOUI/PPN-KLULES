#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "lulesh-init.h"
#include "lulesh-geometry.h"
#include "lulesh-comm.h"

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Int_t myRank, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
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
   m_myRank   = myRank ;

   ///////////////////////////////
   //   Initialize Sedov Mesh
   ///////////////////////////////

   // construct a uniform box for this processor

   m_colLoc   =   colLoc ;
   m_rowLoc   =   rowLoc ;
   m_planeLoc = planeLoc ;
   
   m_sizeX = edgeElems ;
   m_sizeY = edgeElems ;
   m_sizeZ = edgeElems ;
   m_numElem = edgeElems*edgeElems*edgeElems ;

   m_numNode = edgeNodes*edgeNodes*edgeNodes ;

   m_conn.m_regNumList =
      Kokkos::View<Index_t*, Kokkos::HostSpace>("regNumList", numElem()) ;  // material indexset

   // Elem-centered
   AllocateElemPersistent(numElem()) ;

   // Node-centered
   AllocateNodePersistent(numNode()) ;

   // Pre-allocate gradient and strain temporaries — fixed size, reused every step.
   // allElem includes local elements plus one ghost layer on each of the 6 faces
   // (same layout used by the MonoQ halo exchange).
   {
      Index_t allElem = m_numElem
         + 2*m_sizeX*m_sizeY   // planeMin + planeMax ghost planes
         + 2*m_sizeX*m_sizeZ   // rowMin   + rowMax   ghost rows
         + 2*m_sizeY*m_sizeZ ; // colMin   + colMax   ghost columns
      AllocateGradients(m_numElem, allElem) ;
      AllocateStrains(m_numElem) ;
   }

   SetupCommBuffers(edgeNodes);

   // These Views live in device memory on CUDA builds, so initialize them
   // with deep_copy instead of host-side operator().
   Kokkos::deep_copy(m_elems.m_p,   Real_t(0.0)) ;
   Kokkos::deep_copy(m_elems.m_q,   Real_t(0.0)) ;
   Kokkos::deep_copy(m_elems.m_ss,  Real_t(0.0)) ;
   Kokkos::deep_copy(m_elems.m_v,   Real_t(1.0)) ;
   Kokkos::deep_copy(m_nodes.m_xd,  Real_t(0.0)) ;
   Kokkos::deep_copy(m_nodes.m_yd,  Real_t(0.0)) ;
   Kokkos::deep_copy(m_nodes.m_zd,  Real_t(0.0)) ;
   Kokkos::deep_copy(m_nodes.m_xdd, Real_t(0.0)) ;
   Kokkos::deep_copy(m_nodes.m_ydd, Real_t(0.0)) ;
   Kokkos::deep_copy(m_nodes.m_zdd, Real_t(0.0)) ;

   BuildMesh(nx, edgeNodes, edgeElems);

   SetupThreadSupportStructures();

   // Setup region index sets. For now, these are constant sized
   // throughout the run, but could be changed every cycle to
   // simulate effects of ALE on the lagrange solver
   CreateRegionIndexSets(nr, balance);

   // Pre-allocate EOS temporaries to max region size (avoids ~11,000 malloc/free
   // per run for -s 45 -i 200 with 11 regions).
   {
      Index_t maxRegSize = 0;
      for (Int_t r = 0; r < nr; ++r)
         if (regElemSize(r) > maxRegSize) maxRegSize = regElemSize(r);
      m_elems.m_eosTemps.reserve(maxRegSize);
   }

   // Setup symmetry nodesets
   SetupSymmetryPlanes(edgeNodes);

   // Setup element connectivities
   SetupElementConnectivities(edgeElems);

   // Setup symmetry planes and free surface boundary arrays
   SetupBoundaryConditions(edgeElems);


   // Setup defaults

   // These can be changed (requires recompile) if you want to run
   // with a fixed timestep, or to a different end time, but it's
   // probably easier/better to just run a fixed number of timesteps
   // using the -i flag in 2.x

   dtfixed() = Real_t(-1.0e-6) ; // Negative means use courant condition
   stoptime()  = Real_t(1.0e-2); // *Real_t(edgeElems*tp/45.0) ;

   // Initial conditions
   deltatimemultlb() = Real_t(1.1) ;
   deltatimemultub() = Real_t(1.2) ;
   dtcourant() = Real_t(1.0e+20) ;
   dthydro()   = Real_t(1.0e+20) ;
   dtmax()     = Real_t(1.0e-2) ;
   time()    = Real_t(0.) ;
   cycle()   = Int_t(0) ;

   // initialize field data
   {
      auto x_h         = Kokkos::create_mirror_view(m_nodes.m_x) ;
      auto y_h         = Kokkos::create_mirror_view(m_nodes.m_y) ;
      auto z_h         = Kokkos::create_mirror_view(m_nodes.m_z) ;
      auto nodelist_h  = Kokkos::create_mirror_view(m_conn.m_nodelist) ;
      auto volo_h      = Kokkos::create_mirror_view(m_elems.m_volo) ;
      auto elemMass_h  = Kokkos::create_mirror_view(m_elems.m_elemMass) ;
      auto nodalMass_h = Kokkos::create_mirror_view(m_nodes.m_nodalMass) ;
      auto e_h         = Kokkos::create_mirror_view(m_elems.m_e) ;

      Kokkos::deep_copy(x_h, m_nodes.m_x) ;
      Kokkos::deep_copy(y_h, m_nodes.m_y) ;
      Kokkos::deep_copy(z_h, m_nodes.m_z) ;
      Kokkos::deep_copy(nodelist_h, m_conn.m_nodelist) ;
      Kokkos::deep_copy(nodalMass_h, Real_t(0.0)) ;
      Kokkos::deep_copy(e_h, Real_t(0.0)) ;

      for (Index_t i=0; i<numElem(); ++i) {
         Real_t x_local[8], y_local[8], z_local[8] ;
         const Index_t *elemToNode = nodelist_h.data() + Index_t(8)*i ;
         for (Index_t lnode=0 ; lnode<8 ; ++lnode) {
            Index_t gnode = elemToNode[lnode] ;
            x_local[lnode] = x_h(gnode) ;
            y_local[lnode] = y_h(gnode) ;
            z_local[lnode] = z_h(gnode) ;
         }

         Real_t volume = CalcElemVolume(x_local, y_local, z_local ) ;
         volo_h(i) = volume ;
         elemMass_h(i) = volume ;
         for (Index_t j=0; j<8; ++j) {
            Index_t idx = elemToNode[j] ;
            nodalMass_h(idx) += volume / Real_t(8.0) ;
         }
      }

   // deposit initial energy
   // An energy of 3.948746e+7 is correct for a problem with
   // 45 zones along a side - we need to scale it
   const Real_t ebase = Real_t(3.948746e+7);
   Real_t scale = (nx*m_tp)/Real_t(45.0);
   Real_t einit = ebase*scale*scale*scale;
      if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
         // Dump into the first zone (which we know is in the corner)
         // of the domain that sits at the origin
         e_h(0) = einit;
      }
      // set initial deltatime base on analytic CFL calculation
      deltatime() = (Real_t(.5)*cbrt(volo_h(0)))/sqrt(Real_t(2.0)*einit);

      Kokkos::deep_copy(m_elems.m_volo, volo_h) ;
      Kokkos::deep_copy(m_elems.m_elemMass, elemMass_h) ;
      Kokkos::deep_copy(m_nodes.m_nodalMass, nodalMass_h) ;
      Kokkos::deep_copy(m_elems.m_e, e_h) ;
   }

#if USE_MPI
   // Allocate MPI send/recv buffers (depends on sizeX/Y/Z being set above)
   CommSetup(*this);
#endif

} // End constructor


////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  Index_t meshEdgeElems = m_tp*nx ;

  auto x_h = Kokkos::create_mirror_view(m_nodes.m_x) ;
  auto y_h = Kokkos::create_mirror_view(m_nodes.m_y) ;
  auto z_h = Kokkos::create_mirror_view(m_nodes.m_z) ;
  auto nodelist_h = Kokkos::create_mirror_view(m_conn.m_nodelist) ;

  // initialize nodal coordinates
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

  // embed hexehedral elements in nodal point lattice
  Index_t zidx = 0 ;
  nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
        Index_t *localNode = nodelist_h.data() + Index_t(8)*zidx ;
        localNode[0] = nidx                                       ;
        localNode[1] = nidx                                   + 1 ;
        localNode[2] = nidx                       + edgeNodes + 1 ;
        localNode[3] = nidx                       + edgeNodes     ;
        localNode[4] = nidx + edgeNodes*edgeNodes                 ;
        localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
        localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
        localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
        ++zidx ;
        ++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }

  Kokkos::deep_copy(m_nodes.m_x, x_h) ;
  Kokkos::deep_copy(m_nodes.m_y, y_h) ;
  Kokkos::deep_copy(m_nodes.m_z, z_h) ;
  Kokkos::deep_copy(m_conn.m_nodelist, nodelist_h) ;
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupThreadSupportStructures()
{
  // The stress gather path unconditionally reads these CSR-style adjacency
  // tables, so they must exist even when OpenMP is configured with one thread.
  auto nodelist_h = Kokkos::create_mirror_view(m_conn.m_nodelist) ;
  Kokkos::deep_copy(nodelist_h, m_conn.m_nodelist) ;
  std::vector<Index_t> nodeElemCount(numNode(), 0) ;

  for (Index_t i=0; i<numElem(); ++i) {
    const Index_t *nl = nodelist_h.data() + Index_t(8)*i ;
    for (Index_t j=0; j < 8; ++j) {
      ++(nodeElemCount[nl[j]] );
    }
  }

  m_conn.m_nodeElemStart = Kokkos::View<Index_t*>("nodeElemStart", numNode()+1) ;
  auto nodeElemStart_h = Kokkos::create_mirror_view(m_conn.m_nodeElemStart) ;

  nodeElemStart_h(0) = 0;

  for (Index_t i=1; i <= numNode(); ++i) {
    nodeElemStart_h(i) = nodeElemStart_h(i-1) + nodeElemCount[i-1] ;
  }

  m_conn.m_nodeElemCornerList =
     Kokkos::View<Index_t*>("nodeElemCornerList", nodeElemStart_h(numNode()));
  auto nodeElemCornerList_h = Kokkos::create_mirror_view(m_conn.m_nodeElemCornerList) ;

  for (Index_t i=0; i < numNode(); ++i) {
    nodeElemCount[i] = 0;
  }

  for (Index_t i=0; i < numElem(); ++i) {
    const Index_t *nl = nodelist_h.data() + Index_t(8)*i ;
    for (Index_t j=0; j < 8; ++j) {
      Index_t m = nl[j];
      Index_t k = i*8 + j ;
      Index_t offset = nodeElemStart_h(m) + nodeElemCount[m] ;
      nodeElemCornerList_h(offset) = k;
      ++(nodeElemCount[m]) ;
    }
  }

  Index_t clSize = nodeElemStart_h(numNode()) ;
  for (Index_t i=0; i < clSize; ++i) {
    Index_t clv = nodeElemCornerList_h(i) ;
    if ((clv < 0) || (clv > numElem()*8)) {
      fprintf(stderr,
              "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
      exit(-1);
    }
  }

  Kokkos::deep_copy(m_conn.m_nodeElemStart, nodeElemStart_h) ;
  Kokkos::deep_copy(m_conn.m_nodeElemCornerList, nodeElemCornerList_h) ;
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // assume communication to 6 neighbors by default 
  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

  // Boundary nodesets
  if (m_colLoc == 0)
    m_nodes.m_symmX = Kokkos::View<Index_t*>("symmX", edgeNodes*edgeNodes);
  if (m_rowLoc == 0)
    m_nodes.m_symmY = Kokkos::View<Index_t*>("symmY", edgeNodes*edgeNodes);
  if (m_planeLoc == 0)
    m_nodes.m_symmZ = Kokkos::View<Index_t*>("symmZ", edgeNodes*edgeNodes);
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
   srand(0);
   const Index_t myRank = m_myRank;
   this->numReg() = nr;
   m_conn.m_regElemSize =
      Kokkos::View<Index_t*, Kokkos::HostSpace>("regElemSize", numReg());
   m_conn.m_regElemlist.resize(numReg());
   Index_t nextIndex = 0;
   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(numReg() == 1) {
      while (nextIndex < numElem()) {
	 this->regNumList(nextIndex) = 1;
         nextIndex++;
      }
      regElemSize(0) = 0;
   }
   //If we have more than one region distribute the elements.
   else {
      Int_t regionNum;
      Int_t regionVar;
      Int_t lastReg = -1;
      Int_t binSize;
      Index_t elements;
      Index_t runto = 0;
      Int_t costDenominator = 0;
      std::vector<Int_t> regBinEnd(numReg());
      //Determine the relative weights of all the regions.  This is based off the -b flag.  Balance is the value passed into b.  
      for (Index_t i=0 ; i<numReg() ; ++i) {
         regElemSize(i) = 0;
	 costDenominator += pow((i+1), balance);  //Total sum of all regions weights
	 regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < numElem()) {
	 //pick the region
	 regionVar = rand() % costDenominator;
	 Index_t i = 0;
         while(regionVar >= regBinEnd[i])
	    i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions this makes each domain have a different region with 
         //the highest representation
	 regionNum = ((i + myRank) % numReg()) + 1;
	 // make sure we don't pick the same region twice in a row
         while(regionNum == lastReg) {
	    regionVar = rand() % costDenominator;
	    i = 0;
            while(regionVar >= regBinEnd[i])
	       i++;
	    regionNum = ((i + myRank) % numReg()) + 1;
         }
	 //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
	 if(binSize < 773) {
	   elements = rand() % 15 + 1;
	 }
	 else if(binSize < 937) {
	   elements = rand() % 16 + 16;
	 }
	 else if(binSize < 970) {
	   elements = rand() % 32 + 32;
	 }
	 else if(binSize < 974) {
	   elements = rand() % 64 + 64;
	 } 
	 else if(binSize < 978) {
	   elements = rand() % 128 + 128;
	 }
	 else if(binSize < 981) {
	   elements = rand() % 256 + 256;
	 }
	 else
	    elements = rand() % 1537 + 512;
	 runto = elements + nextIndex;
	 //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < numElem()) {
	    this->regNumList(nextIndex) = regionNum;
	    nextIndex++;
	 }
	 lastReg = regionNum;
      } 
   }
   // Convert regNumList to region index sets
   // First, count size of each region 
   for (Index_t i=0 ; i<numElem() ; ++i) {
      int r = this->regNumList(i)-1; // region index == regnum-1
      regElemSize(r)++;
   }
   // Second, allocate each region index set
   for (Index_t i=0 ; i<numReg() ; ++i) {
      m_conn.m_regElemlist[i].resize(regElemSize(i));
      regElemSize(i) = 0;
   }
   // Third, fill index sets
   for (Index_t i=0 ; i<numElem() ; ++i) {
      Index_t r = regNumList(i)-1;       // region index == regnum-1
      Index_t regndx = regElemSize(r)++; // Note increment
      regElemlist(r,regndx) = i;
   }
   
}

/////////////////////////////////////////////////////////////
void 
Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
  auto symmX_h = Kokkos::create_mirror_view(m_nodes.m_symmX) ;
  auto symmY_h = Kokkos::create_mirror_view(m_nodes.m_symmY) ;
  auto symmZ_h = Kokkos::create_mirror_view(m_nodes.m_symmZ) ;

  Index_t nidx = 0 ;
  for (Index_t i=0; i<edgeNodes; ++i) {
    Index_t planeInc = i*edgeNodes*edgeNodes ;
    Index_t rowInc   = i*edgeNodes ;
    for (Index_t j=0; j<edgeNodes; ++j) {
      if (m_planeLoc == 0) {
        symmZ_h(nidx) = rowInc   + j ;
      }
      if (m_rowLoc == 0) {
        symmY_h(nidx) = planeInc + j ;
      }
      if (m_colLoc == 0) {
        symmX_h(nidx) = planeInc + j*edgeNodes ;
      }
      ++nidx ;
    }
  }

  if (m_planeLoc == 0) Kokkos::deep_copy(m_nodes.m_symmZ, symmZ_h) ;
  if (m_rowLoc == 0)   Kokkos::deep_copy(m_nodes.m_symmY, symmY_h) ;
  if (m_colLoc == 0)   Kokkos::deep_copy(m_nodes.m_symmX, symmX_h) ;
}



/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Int_t edgeElems)
{
   auto lxim_h   = Kokkos::create_mirror_view(m_conn.m_lxim) ;
   auto lxip_h   = Kokkos::create_mirror_view(m_conn.m_lxip) ;
   auto letam_h  = Kokkos::create_mirror_view(m_conn.m_letam) ;
   auto letap_h  = Kokkos::create_mirror_view(m_conn.m_letap) ;
   auto lzetam_h = Kokkos::create_mirror_view(m_conn.m_lzetam) ;
   auto lzetap_h = Kokkos::create_mirror_view(m_conn.m_lzetap) ;

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

   Kokkos::deep_copy(m_conn.m_lxim, lxim_h) ;
   Kokkos::deep_copy(m_conn.m_lxip, lxip_h) ;
   Kokkos::deep_copy(m_conn.m_letam, letam_h) ;
   Kokkos::deep_copy(m_conn.m_letap, letap_h) ;
   Kokkos::deep_copy(m_conn.m_lzetam, lzetam_h) ;
   Kokkos::deep_copy(m_conn.m_lzetap, lzetap_h) ;
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Int_t edgeElems) 
{
  auto lxim_h   = Kokkos::create_mirror_view(m_conn.m_lxim) ;
  auto lxip_h   = Kokkos::create_mirror_view(m_conn.m_lxip) ;
  auto letam_h  = Kokkos::create_mirror_view(m_conn.m_letam) ;
  auto letap_h  = Kokkos::create_mirror_view(m_conn.m_letap) ;
  auto lzetam_h = Kokkos::create_mirror_view(m_conn.m_lzetam) ;
  auto lzetap_h = Kokkos::create_mirror_view(m_conn.m_lzetap) ;
  auto elemBC_h = Kokkos::create_mirror_view(m_conn.m_elemBC) ;

  Kokkos::deep_copy(lxim_h, m_conn.m_lxim) ;
  Kokkos::deep_copy(lxip_h, m_conn.m_lxip) ;
  Kokkos::deep_copy(letam_h, m_conn.m_letam) ;
  Kokkos::deep_copy(letap_h, m_conn.m_letap) ;
  Kokkos::deep_copy(lzetam_h, m_conn.m_lzetam) ;
  Kokkos::deep_copy(lzetap_h, m_conn.m_lzetap) ;

  Index_t ghostIdx[6] ;  // offsets to ghost locations

  // set up boundary condition information
  for (Index_t i=0; i<numElem(); ++i) {
     elemBC_h(i) = Int_t(0) ;
  }

  for (Index_t i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = numElem() ;
  if (m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += sizeY()*sizeZ() ;
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  // symmetry plane or free surface BCs 
  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (m_planeLoc == 0) {
        elemBC_h(rowInc+j) |= ZETA_M_SYMM ;
      }
      else {
        elemBC_h(rowInc+j) |= ZETA_M_COMM ;
        lzetam_h(rowInc+j) = ghostIdx[0] + rowInc + j ;
      }

      if (m_planeLoc == m_tp-1) {
        elemBC_h(rowInc+j+numElem()-edgeElems*edgeElems) |= ZETA_P_FREE;
      }
      else {
        elemBC_h(rowInc+j+numElem()-edgeElems*edgeElems) |= ZETA_P_COMM ;
        lzetap_h(rowInc+j+numElem()-edgeElems*edgeElems) =
           ghostIdx[1] + rowInc + j ;
      }

      if (m_rowLoc == 0) {
        elemBC_h(planeInc+j) |= ETA_M_SYMM ;
      }
      else {
        elemBC_h(planeInc+j) |= ETA_M_COMM ;
        letam_h(planeInc+j) = ghostIdx[2] + rowInc + j ;
      }

      if (m_rowLoc == m_tp-1) {
        elemBC_h(planeInc+j+edgeElems*edgeElems-edgeElems) |= ETA_P_FREE ;
      }
      else {
        elemBC_h(planeInc+j+edgeElems*edgeElems-edgeElems) |= ETA_P_COMM ;
        letap_h(planeInc+j+edgeElems*edgeElems-edgeElems) =
           ghostIdx[3] +  rowInc + j ;
      }

      if (m_colLoc == 0) {
        elemBC_h(planeInc+j*edgeElems) |= XI_M_SYMM ;
      }
      else {
        elemBC_h(planeInc+j*edgeElems) |= XI_M_COMM ;
        lxim_h(planeInc+j*edgeElems) = ghostIdx[4] + rowInc + j ;
      }

      if (m_colLoc == m_tp-1) {
        elemBC_h(planeInc+j*edgeElems+edgeElems-1) |= XI_P_FREE ;
      }
      else {
        elemBC_h(planeInc+j*edgeElems+edgeElems-1) |= XI_P_COMM ;
        lxip_h(planeInc+j*edgeElems+edgeElems-1) =
           ghostIdx[5] + rowInc + j ;
      }
    }
  }

  Kokkos::deep_copy(m_conn.m_lxim, lxim_h) ;
  Kokkos::deep_copy(m_conn.m_lxip, lxip_h) ;
  Kokkos::deep_copy(m_conn.m_letam, letam_h) ;
  Kokkos::deep_copy(m_conn.m_letap, letap_h) ;
  Kokkos::deep_copy(m_conn.m_lzetam, lzetam_h) ;
  Kokkos::deep_copy(m_conn.m_lzetap, lzetap_h) ;
  Kokkos::deep_copy(m_conn.m_elemBC, elemBC_h) ;
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;
   
   // Assume cube processor layout for now 
   testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
      exit(-1);
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("Only float and double are supported.\n");
      exit(-1);
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   // temporary test
   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
      exit(-1);
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
