/*
 * lulesh-comm.cc — halo-exchange routines backed by kokkos-comm.
 *
 * Communication pattern: each rank has up to 6 face-adjacent neighbors in a
 * 3-D Cartesian decomposition (planeMin/planeMax, rowMin/rowMax, colMin/colMax).
 * Only face exchanges are performed (no edge/corner).
 *
 * Two exchange types:
 *   MSG_COMM_SBN  — subdomain boundary node forces (scatter-add)
 *   MSG_MONOQ     — velocity gradients for Q (direct ghost overwrite)
 *
 * Current kokkos-comm point-to-point APIs use a fixed internal tag, so we
 * isolate the two phases with duplicated communicators instead of raw MPI tags.
 */

#include "lulesh-comm.h"
#include "lulesh-profile.h"
#include "lulesh-runtime.h"

#if USE_MPI

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Definition of the global comm state (declared extern in lulesh.h)
DomainComm g_comm ;

namespace {

using HostBuffer = DomainComm::buffer_type ;
using HostSlice = Kokkos::View<Real_t*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>> ;
using Request = DomainComm::request_type ;
using Communicator = DomainComm::communicator_type ;

struct MonoQExchangeLayout {
   std::array<int, 6> hasNeighbor {} ;
   std::array<Index_t, 6> faceSizes {} ;
   std::array<Index_t, 6> faceOffsets {} ;
   std::array<Index_t, 6> ghostBase {} ;
   Index_t planeGhostBase = 0 ;
   Index_t planeGhostCount = 0 ;
   Index_t rowColGhostBase = 0 ;
   Index_t rowColGhostCount = 0 ;
} ;

[[noreturn]] auto CommAbort(const char* message) -> void
{
   std::fprintf(stderr, "LULESH communication error: %s\n", message) ;
   std::abort() ;
}

static int NeighborRank(Domain& domain, int dc, int dr, int dp)
{
   int col   = domain.colLoc()   + dc ;
   int row   = domain.rowLoc()   + dr ;
   int plane = domain.planeLoc() + dp ;
   int tp    = domain.tp() ;
   return plane * tp * tp + row * tp + col ;
}

auto PhaseCommunicator(int msgType) -> Communicator&
{
   if (msgType == MSG_COMM_SBN) {
      if (!g_comm.sbnComm) {
         CommAbort("SBN communicator requested before CommSetup") ;
      }
      return *g_comm.sbnComm ;
   }
   if (msgType == MSG_MONOQ) {
      if (!g_comm.monoqComm) {
         CommAbort("MONOQ communicator requested before CommSetup") ;
      }
      return *g_comm.monoqComm ;
   }
   CommAbort("unknown communication phase") ;
}

auto MakeSlice(HostBuffer& buffer, Index_t offset, Index_t count) -> HostSlice
{
   return HostSlice(buffer.data() + offset, count) ;
}

auto ResetRequests(std::array<Request, 6>& requests) -> void
{
   for (auto& req : requests) {
      req = Request{} ;
   }
}

auto WaitFaceSends(const int hasNeighbor[6]) -> void
{
   for (int face = 0; face < 6; ++face) {
      if (hasNeighbor[face]) {
         KokkosComm::wait(g_comm.sendRequest[face]) ;
      }
   }
}

auto BuildMonoQExchangeLayout(Domain& domain) -> MonoQExchangeLayout
{
   MonoQExchangeLayout layout ;

   Index_t nx = domain.sizeX() ;
   Index_t ny = domain.sizeY() ;
   Index_t nz = domain.sizeZ() ;
   Index_t numElem = domain.numElem() ;

   layout.hasNeighbor = {
      domain.planeMin(), domain.planeMax(),
      domain.rowMin(),   domain.rowMax(),
      domain.colMin(),   domain.colMax()
   } ;

   layout.faceSizes = {
      nx*ny, nx*ny,
      nx*nz, nx*nz,
      ny*nz, ny*nz
   } ;

   Index_t offset = 0 ;
   for (std::size_t face = 0; face < layout.faceSizes.size(); ++face) {
      layout.faceOffsets[face] = offset ;
      offset += 3 * layout.faceSizes[face] ;
   }

   Index_t pid = numElem ;
   for (std::size_t face = 0; face < layout.faceSizes.size(); ++face) {
      layout.ghostBase[face] = layout.hasNeighbor[face] ? pid : -1 ;
      pid += layout.faceSizes[face] ;
   }

   layout.planeGhostBase = numElem ;
   layout.planeGhostCount = 2 * nx * ny ;
   layout.rowColGhostBase = numElem + layout.planeGhostCount ;
   layout.rowColGhostCount = 2 * nx * nz + 2 * ny * nz ;
   return layout ;
}

auto WaitAndUnpackMonoQFaces(DomainComm::buffer_type dxi_h,
                             DomainComm::buffer_type deta_h,
                             DomainComm::buffer_type dzeta_h,
                             const MonoQExchangeLayout& layout,
                             int faceBegin,
                             int faceEnd) -> void
{
   for (int face = faceBegin; face < faceEnd; ++face) {
      if (!layout.hasNeighbor[face]) {
         continue ;
      }

      auto waitStart = std::chrono::steady_clock::now() ;
      {
         ProfileScope commMonoqWaitTimer(ProfileTimer::commmonoq_wait, false) ;
         KokkosComm::wait(g_comm.recvRequest[face]) ;
      }
      auto waitElapsed =
         std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart) ;
      ProfileAddMonoqRecvWaitFaceSample(static_cast<std::size_t>(face),
                                        waitElapsed.count()) ;
      ProfileCompleteMonoqRecvFace(static_cast<std::size_t>(face)) ;

      Real_t* buf = g_comm.recvBuf.data() + layout.faceOffsets[face] ;
      Index_t n = layout.faceSizes[face] ;
      Index_t base = layout.ghostBase[face] ;

      {
         ProfileScope commMonoqUnpackTimer(ProfileTimer::commmonoq_unpack, false) ;
         for (Index_t e = 0; e < n; ++e) {
            dxi_h(base + e)   = buf[3*e    ] ;
            deta_h(base + e)  = buf[3*e + 1] ;
            dzeta_h(base + e) = buf[3*e + 2] ;
         }
      }
   }
}

auto PushMonoQGhostRange(Domain& domain,
                         DomainComm::buffer_type dxi_h,
                         DomainComm::buffer_type deta_h,
                         DomainComm::buffer_type dzeta_h,
                         Index_t begin,
                         Index_t count) -> void
{
   if (count <= 0) {
      return ;
   }

   auto range = std::pair<Index_t, Index_t>(begin, begin + count) ;

   {
      ProfileScope commMonoqHostPushTimer(ProfileTimer::commmonoq_host_push) ;
      Kokkos::deep_copy(Kokkos::subview(domain.m_elems.m_delv_xi, range),
                        Kokkos::subview(dxi_h, range)) ;
      Kokkos::deep_copy(Kokkos::subview(domain.m_elems.m_delv_eta, range),
                        Kokkos::subview(deta_h, range)) ;
      Kokkos::deep_copy(Kokkos::subview(domain.m_elems.m_delv_zeta, range),
                        Kokkos::subview(dzeta_h, range)) ;
   }
}

auto WaitMonoQSends(const MonoQExchangeLayout& layout, int faceBegin, int faceEnd) -> void
{
   for (int face = faceBegin; face < faceEnd; ++face) {
      if (!layout.hasNeighbor[face]) {
         continue ;
      }

      auto waitStart = std::chrono::steady_clock::now() ;
      {
         ProfileScope commMonoqSendWaitTimer(ProfileTimer::commmonoq_send_wait, false) ;
         KokkosComm::wait(g_comm.sendRequest[face]) ;
      }
      auto waitElapsed =
         std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart) ;
      ProfileAddMonoqSendWaitFaceSample(static_cast<std::size_t>(face),
                                        waitElapsed.count()) ;
      ProfileCompleteMonoqSendFace(static_cast<std::size_t>(face)) ;
   }
}

} // namespace

void CommSetup(Domain& domain)
{
   Index_t nodePlane = (domain.sizeX() + 1) * (domain.sizeY() + 1) ;
   Index_t nodeRow   = (domain.sizeX() + 1) * (domain.sizeZ() + 1) ;
   Index_t nodeCol   = (domain.sizeY() + 1) * (domain.sizeZ() + 1) ;
   Index_t maxFace = std::max({nodePlane, nodeRow, nodeCol}) ;
   Index_t bufSize = 3 * 6 * maxFace ;

   g_comm.sendBuf = HostBuffer("comm_send_buf", bufSize) ;
   g_comm.recvBuf = HostBuffer("comm_recv_buf", bufSize) ;
   ResetRequests(g_comm.recvRequest) ;
   ResetRequests(g_comm.sendRequest) ;

   // Pre-allocate allreduce staging buffers (O-A: scalar pair removed).
   g_comm.dt_send_pair = HostBuffer("dt_send_pair", 2) ;
   g_comm.dt_recv_pair = HostBuffer("dt_recv_pair", 2) ;

   // O-B: pre-allocate allElem host mirrors for MonoQ halo exchange.
   // allElem = numElem + ghost slots on all 6 faces (matches AllocateGradients).
   Index_t nx = domain.sizeX(), ny = domain.sizeY(), nz = domain.sizeZ() ;
   Index_t allElem = domain.numElem()
      + 2*nx*ny   // planeMin + planeMax ghost planes
      + 2*nx*nz   // rowMin   + rowMax   ghost rows
      + 2*ny*nz ; // colMin   + colMax   ghost columns
   g_comm.monoq_dxi_h   = HostBuffer("monoq_dxi_h",   allElem) ;
   g_comm.monoq_deta_h  = HostBuffer("monoq_deta_h",  allElem) ;
   g_comm.monoq_dzeta_h = HostBuffer("monoq_dzeta_h", allElem) ;

   // P3: pre-allocate numNode host mirrors for SBN (node-force) halo exchange.
   Index_t numNode = (nx+1) * (ny+1) * (nz+1) ;
   g_comm.sbn_fx_h = HostBuffer("sbn_fx_h", numNode) ;
   g_comm.sbn_fy_h = HostBuffer("sbn_fy_h", numNode) ;
   g_comm.sbn_fz_h = HostBuffer("sbn_fz_h", numNode) ;

   // O-E/T2: build staged MonoQ overlap index lists.
   // Interior: no communicating neighbor on any face.
   // Row/col boundary: reads only row/col ghost data, so it can run after the
   //                   row/col unpack stage while plane receives still fly.
   // Plane boundary: any plane ghost dependency keeps the element in the final
   //                 stage because plane traffic is the long pole.
   {
      std::vector<Index_t> interior_h, rowcol_boundary_h, plane_boundary_h ;
      interior_h.reserve(domain.numElem()) ;
      rowcol_boundary_h.reserve(domain.numElem()) ;
      plane_boundary_h.reserve(domain.numElem()) ;

      for (Index_t k = 0; k < nz; ++k) {
         bool zComm = (k == 0 && domain.planeMin()) ||
                      (k == nz-1 && domain.planeMax()) ;
         for (Index_t j = 0; j < ny; ++j) {
            bool yComm = (j == 0 && domain.rowMin()) ||
                         (j == ny-1 && domain.rowMax()) ;
            for (Index_t i = 0; i < nx; ++i) {
               bool xComm = (i == 0 && domain.colMin()) ||
                            (i == nx-1 && domain.colMax()) ;
               Index_t e = k*ny*nx + j*nx + i ;
               if (zComm) {
                  plane_boundary_h.push_back(e) ;
               }
               else if (xComm || yComm) {
                  rowcol_boundary_h.push_back(e) ;
               }
               else {
                  interior_h.push_back(e) ;
               }
            }
         }
      }

      g_comm.interiorElems =
         Kokkos::View<Index_t*>("interior_elems", interior_h.size()) ;
      g_comm.rowColBoundaryElems =
         Kokkos::View<Index_t*>("rowcol_boundary_elems", rowcol_boundary_h.size()) ;
      g_comm.planeBoundaryElems =
         Kokkos::View<Index_t*>("plane_boundary_elems", plane_boundary_h.size()) ;

      auto int_h    = Kokkos::create_mirror_view(g_comm.interiorElems) ;
      auto rowcol_h = Kokkos::create_mirror_view(g_comm.rowColBoundaryElems) ;
      auto plane_h  = Kokkos::create_mirror_view(g_comm.planeBoundaryElems) ;
      for (std::size_t ii = 0; ii < interior_h.size(); ++ii)
         int_h(ii) = interior_h[ii] ;
      for (std::size_t ii = 0; ii < rowcol_boundary_h.size(); ++ii)
         rowcol_h(ii) = rowcol_boundary_h[ii] ;
      for (std::size_t ii = 0; ii < plane_boundary_h.size(); ++ii)
         plane_h(ii) = plane_boundary_h[ii] ;
      Kokkos::deep_copy(g_comm.interiorElems, int_h) ;
      Kokkos::deep_copy(g_comm.rowColBoundaryElems, rowcol_h) ;
      Kokkos::deep_copy(g_comm.planeBoundaryElems, plane_h) ;
   }

   auto sbnComm = WorldComm().duplicate() ;
   auto monoqComm = WorldComm().duplicate() ;
   if (!sbnComm || !monoqComm) {
      CommAbort("failed to duplicate phase communicators") ;
   }

   g_comm.sbnComm = std::move(*sbnComm) ;
   g_comm.monoqComm = std::move(*monoqComm) ;
}

void CommTeardown()
{
   ResetRequests(g_comm.recvRequest) ;
   ResetRequests(g_comm.sendRequest) ;
   g_comm.sbnComm.reset() ;
   g_comm.monoqComm.reset() ;
   g_comm.sendBuf = HostBuffer() ;
   g_comm.recvBuf = HostBuffer() ;
   g_comm.dt_send_pair   = HostBuffer() ;
   g_comm.dt_recv_pair   = HostBuffer() ;
   g_comm.monoq_dxi_h    = HostBuffer() ;
   g_comm.monoq_deta_h   = HostBuffer() ;
   g_comm.monoq_dzeta_h  = HostBuffer() ;
   g_comm.sbn_fx_h       = HostBuffer() ;
   g_comm.sbn_fy_h       = HostBuffer() ;
   g_comm.sbn_fz_h       = HostBuffer() ;
   g_comm.interiorElems       = Kokkos::View<Index_t*>() ;
   g_comm.rowColBoundaryElems = Kokkos::View<Index_t*>() ;
   g_comm.planeBoundaryElems  = Kokkos::View<Index_t*>() ;
}

void CommRecv(Domain& domain, int msgType, Index_t fieldCount,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv)
{
   ResetRequests(g_comm.recvRequest) ;
   if (!doRecv) return ;

   auto& comm = PhaseCommunicator(msgType) ;

   int hasNeighbor[6] = {
      domain.planeMin(), domain.planeMax(),
      domain.rowMin(),   domain.rowMax(),
      domain.colMin(),   domain.colMax()
   } ;
   // {dc, dr, dp} offset to reach each neighbor
   int offsets[6][3] = {
      { 0,  0, -1}, { 0,  0, +1},
      { 0, -1,  0}, { 0, +1,  0},
      {-1,  0,  0}, {+1,  0,  0},
   } ;
   Index_t faceSizes[6] = {
      dx*dy, dx*dy,
      dx*dz, dx*dz,
      dy*dz, dy*dz
   } ;

   {
      ProfileScope commRecvPostTimer(ProfileTimer::commrecv_post, false) ;
      Index_t offset = 0 ;
      for (int i = 0; i < 6; ++i) {
         if (hasNeighbor[i]) {
            int srcRank = NeighborRank(domain,
                                       offsets[i][0], offsets[i][1], offsets[i][2]) ;
            Index_t count = fieldCount * faceSizes[i] ;
            auto slice = MakeSlice(g_comm.recvBuf, offset, count) ;
            g_comm.recvRequest[i] = KokkosComm::recv(comm, slice, srcRank) ;
            if (msgType == MSG_MONOQ) {
               ProfileMarkMonoqRecvPost(static_cast<std::size_t>(i)) ;
            }
         }
         offset += fieldCount * faceSizes[i] ;
      }
   }
}

void CommSend(Domain& domain, int msgType,
              Index_t fieldCount,
              std::vector<Real_t*>& fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend)
{
   ResetRequests(g_comm.sendRequest) ;
   if (!doSend) return ;

   auto& comm = PhaseCommunicator(msgType) ;

   // Use dx/dy/dz as BOTH loop bounds and array strides.
   // For elements: dx=nx, dy=ny, dz=nz  → stride is nx, ny per plane
   // For nodes:    dx=EN, dy=EN, dz=EN  → stride is EN, EN per plane
   // The caller is responsible for passing the correct values.

   int hasNeighbor[6] = {
      domain.planeMin(), domain.planeMax(),
      domain.rowMin(),   domain.rowMax(),
      domain.colMin(),   domain.colMax()
   } ;
   int offsets[6][3] = {
      { 0,  0, -1}, { 0,  0, +1},
      { 0, -1,  0}, { 0, +1,  0},
      {-1,  0,  0}, {+1,  0,  0},
   } ;
   Index_t faceSizes[6] = {
      dx*dy, dx*dy,
      dx*dz, dx*dz,
      dy*dz, dy*dz
   } ;

   Index_t offset = 0 ;
   for (int face = 0; face < 6; ++face) {
      if (!hasNeighbor[face]) {
         offset += fieldCount * faceSizes[face] ;
         continue ;
      }

      {
         ProfileScope commSendPackTimer(ProfileTimer::commsend_pack, false) ;
         Real_t* buf = g_comm.sendBuf.data() + offset ;
         Index_t idx = 0 ;

         if (face == 0) {
            // planeMin: zeta=0 plane (k=0); array stride dx*dy per z-slice
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t j = 0; j < dy; ++j)
                  for (Index_t i = 0; i < dx; ++i)
                     buf[idx++] = fieldData[fld][j*dx + i] ;
         } else if (face == 1) {
            // planeMax: last z-slice
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t j = 0; j < dy; ++j)
                  for (Index_t i = 0; i < dx; ++i)
                     buf[idx++] = fieldData[fld][(dz-1)*dy*dx + j*dx + i] ;
         } else if (face == 2) {
            // rowMin: eta=0 (j=0)
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t k = 0; k < dz; ++k)
                  for (Index_t i = 0; i < dx; ++i)
                     buf[idx++] = fieldData[fld][k*dy*dx + i] ;
         } else if (face == 3) {
            // rowMax: j=dy-1
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t k = 0; k < dz; ++k)
                  for (Index_t i = 0; i < dx; ++i)
                     buf[idx++] = fieldData[fld][k*dy*dx + (dy-1)*dx + i] ;
         } else if (face == 4) {
            // colMin: xi=0 (i=0)
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t k = 0; k < dz; ++k)
                  for (Index_t j = 0; j < dy; ++j)
                     buf[idx++] = fieldData[fld][k*dy*dx + j*dx] ;
         } else {
            // colMax: i=dx-1
            for (Index_t fld = 0; fld < fieldCount; ++fld)
               for (Index_t k = 0; k < dz; ++k)
                  for (Index_t j = 0; j < dy; ++j)
                     buf[idx++] = fieldData[fld][k*dy*dx + j*dx + (dx-1)] ;
         }
      }

      int dstRank = NeighborRank(domain,
                                  offsets[face][0], offsets[face][1], offsets[face][2]) ;
      {
         ProfileScope commSendPostTimer(ProfileTimer::commsend_post, false) ;
         auto slice = MakeSlice(g_comm.sendBuf, offset, fieldCount * faceSizes[face]) ;
         g_comm.sendRequest[face] = KokkosComm::send(comm, slice, dstRank) ;
         if (msgType == MSG_MONOQ) {
            ProfileMarkMonoqSendPost(static_cast<std::size_t>(face)) ;
         }
      }

      offset += fieldCount * faceSizes[face] ;
   }
}

void CommSBN(Domain& domain, int fieldCount,
             std::vector<Real_t*>& fieldData)
{
   Index_t nx = domain.sizeX() ;
   Index_t ny = domain.sizeY() ;
   Index_t nz = domain.sizeZ() ;
   Index_t nxn = nx + 1 ;
   Index_t nyn = ny + 1 ;
   Index_t nzn = nz + 1 ;

   int hasNeighbor[6] = {
      domain.planeMin(), domain.planeMax(),
      domain.rowMin(),   domain.rowMax(),
      domain.colMin(),   domain.colMax()
   } ;

   Index_t faceSizesNode[6] = {
      nxn*nyn, nxn*nyn,
      nxn*nzn, nxn*nzn,
      nyn*nzn, nyn*nzn
   } ;

   Index_t offset = 0 ;
   for (int face = 0; face < 6; ++face) {
      if (hasNeighbor[face]) {
         {
            ProfileScope commSbnWaitTimer(ProfileTimer::commsbn_wait, false) ;
            KokkosComm::wait(g_comm.recvRequest[face]) ;
         }

         Real_t* buf = g_comm.recvBuf.data() + offset ;
         {
            ProfileScope commSbnUnpackTimer(ProfileTimer::commsbn_unpack_add, false) ;
            Index_t idx = 0 ;

            if (face == 0) {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t j = 0; j < nyn; ++j)
                     for (Index_t i = 0; i < nxn; ++i)
                        fieldData[fld][j*nxn + i] += buf[idx++] ;
            } else if (face == 1) {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t j = 0; j < nyn; ++j)
                     for (Index_t i = 0; i < nxn; ++i)
                        fieldData[fld][nz*nyn*nxn + j*nxn + i] += buf[idx++] ;
            } else if (face == 2) {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t k = 0; k < nzn; ++k)
                     for (Index_t i = 0; i < nxn; ++i)
                        fieldData[fld][k*nyn*nxn + i] += buf[idx++] ;
            } else if (face == 3) {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t k = 0; k < nzn; ++k)
                     for (Index_t i = 0; i < nxn; ++i)
                        fieldData[fld][k*nyn*nxn + ny*nxn + i] += buf[idx++] ;
            } else if (face == 4) {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t k = 0; k < nzn; ++k)
                     for (Index_t j = 0; j < nyn; ++j)
                        fieldData[fld][k*nyn*nxn + j*nxn] += buf[idx++] ;
            } else {
               for (Index_t fld = 0; fld < fieldCount; ++fld)
                  for (Index_t k = 0; k < nzn; ++k)
                     for (Index_t j = 0; j < nyn; ++j)
                        fieldData[fld][k*nyn*nxn + j*nxn + nx] += buf[idx++] ;
            }
         }
      }
      offset += fieldCount * faceSizesNode[face] ;
   }

   WaitFaceSends(hasNeighbor) ;
}

void CommMonoQRowCol(Domain& domain,
                     DomainComm::buffer_type dxi_h,
                     DomainComm::buffer_type deta_h,
                     DomainComm::buffer_type dzeta_h)
{
   auto layout = BuildMonoQExchangeLayout(domain) ;
   bool hasRowColNeighbor =
      layout.hasNeighbor[2] || layout.hasNeighbor[3] ||
      layout.hasNeighbor[4] || layout.hasNeighbor[5] ;
   if (!hasRowColNeighbor) {
      return ;
   }

   // dxi_h/deta_h/dzeta_h already contain the host copy of local data from the
   // earlier deep_copy in CalcQForElems; only the row/col ghost slots need to
   // be refreshed here.
   WaitAndUnpackMonoQFaces(dxi_h, deta_h, dzeta_h, layout, 2, 6) ;
   PushMonoQGhostRange(domain, dxi_h, deta_h, dzeta_h,
                       layout.rowColGhostBase, layout.rowColGhostCount) ;
}

void CommMonoQPlane(Domain& domain,
                    DomainComm::buffer_type dxi_h,
                    DomainComm::buffer_type deta_h,
                    DomainComm::buffer_type dzeta_h)
{
   auto layout = BuildMonoQExchangeLayout(domain) ;
   bool hasPlaneNeighbor = layout.hasNeighbor[0] || layout.hasNeighbor[1] ;

   if (hasPlaneNeighbor) {
      WaitAndUnpackMonoQFaces(dxi_h, deta_h, dzeta_h, layout, 0, 2) ;
      PushMonoQGhostRange(domain, dxi_h, deta_h, dzeta_h,
                          layout.planeGhostBase, layout.planeGhostCount) ;
   }

   WaitMonoQSends(layout, 0, 6) ;
}

#endif  // USE_MPI
