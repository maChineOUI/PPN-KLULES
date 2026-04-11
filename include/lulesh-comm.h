#pragma once

#include "lulesh.h"

#if USE_MPI

// Allocate persistent send/recv buffers and duplicate per-phase communicators.
// maxPlaneSize = sizeX*sizeY, maxEdgeSize = sizeX (largest face/edge).
void CommSetup(Domain& domain);

// Release duplicated communicators before MPI_Finalize and clear buffers.
void CommTeardown();

// Post receives for all 6 face-neighbors before computation begins.
// msgType: MSG_COMM_SBN or MSG_MONOQ
// fieldCount: number of Real_t fields packed per element/node on each face
void CommRecv(Domain& domain, int msgType, Index_t fieldCount,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv);

// Pack boundary data and send to all 6 face-neighbors.
void CommSend(Domain& domain, int msgType,
              Index_t fieldCount,
              std::vector<Real_t*>& fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend);

// Unpack and accumulate (SBN = subdomain boundary nodes: forces).
// Called after CommSend+CommRecv; Waitall then scatter into node arrays.
void CommSBN(Domain& domain, int fieldCount,
             std::vector<Real_t*>& fieldData);

// Unpack row/col monotonic-Q ghost gradients (no accumulation, direct
// overwrite). The caller may then compute elements that only depend on row/col
// ghost data while plane receives are still in flight.
void CommMonoQRowCol(Domain& domain,
                     DomainComm::buffer_type dxi_h,
                     DomainComm::buffer_type deta_h,
                     DomainComm::buffer_type dzeta_h);

// Unpack plane monotonic-Q ghost gradients and complete the remaining send
// waits. Called after the row/col overlap stage, right before plane-dependent
// boundary elements are evaluated.
void CommMonoQPlane(Domain& domain,
                    DomainComm::buffer_type dxi_h,
                    DomainComm::buffer_type deta_h,
                    DomainComm::buffer_type dzeta_h);

#endif  // USE_MPI
