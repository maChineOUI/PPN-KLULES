#include "lulesh-comm.h"

#include <array>
#include <cstdlib>
#include <utility>
#include <vector>

#if USE_MPI

namespace {

using DeviceFieldView = Kokkos::View<Real_t*>;
using HostFieldView =
   decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                std::declval<DeviceFieldView>()));

struct CommFlags {
   bool rowMin;
   bool rowMax;
   bool colMin;
   bool colMax;
   bool planeMin;
   bool planeMax;
};

[[noreturn]] void AbortComm(const char *message)
{
   (void)message;
   MPI_Abort(MPI_COMM_WORLD, -1);
   std::abort();
}

MPI_Datatype MpiRealType()
{
   return (sizeof(Real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
}

CommFlags GetCommFlags(Domain& domain)
{
   return {
      domain.rowLoc() != 0,
      domain.rowLoc() != (domain.tp() - 1),
      domain.colLoc() != 0,
      domain.colLoc() != (domain.tp() - 1),
      domain.planeLoc() != 0,
      domain.planeLoc() != (domain.tp() - 1)
   };
}

DeviceFieldView GetDeviceFieldView(Domain& domain, Domain_member field)
{
   if (field == &Domain::x)        return domain.m_nodes.m_x;
   if (field == &Domain::y)        return domain.m_nodes.m_y;
   if (field == &Domain::z)        return domain.m_nodes.m_z;
   if (field == &Domain::xd)       return domain.m_nodes.m_xd;
   if (field == &Domain::yd)       return domain.m_nodes.m_yd;
   if (field == &Domain::zd)       return domain.m_nodes.m_zd;
   if (field == &Domain::fx)       return domain.m_nodes.m_fx;
   if (field == &Domain::fy)       return domain.m_nodes.m_fy;
   if (field == &Domain::fz)       return domain.m_nodes.m_fz;
   if (field == &Domain::delv_xi)  return domain.m_elems.m_delv_xi;
   if (field == &Domain::delv_eta) return domain.m_elems.m_delv_eta;
   if (field == &Domain::delv_zeta)return domain.m_elems.m_delv_zeta;

   AbortComm("Unsupported communication field");
}

std::vector<HostFieldView>
CreateHostFieldCopies(Domain& domain, Index_t xferFields, Domain_member *fieldData)
{
   std::vector<HostFieldView> fields;
   fields.reserve(static_cast<std::size_t>(xferFields));
   for (Index_t fi = 0; fi < xferFields; ++fi) {
      fields.push_back(
         Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                             GetDeviceFieldView(domain, fieldData[fi])));
   }
   return fields;
}

void CopyFieldsBack(Domain& domain, Index_t xferFields, Domain_member *fieldData,
                    const std::vector<HostFieldView>& fields)
{
   for (Index_t fi = 0; fi < xferFields; ++fi) {
      Kokkos::deep_copy(GetDeviceFieldView(domain, fieldData[fi]), fields[fi]);
   }
}

} // namespace

void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz, bool doRecv, bool planeOnly)
{
   if (domain.numRanks() == 1) {
      return;
   }

   const Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   const Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize();
   Index_t pmsg = 0;
   Index_t emsg = 0;
   Index_t cmsg = 0;
   int myRank = 0;
   const CommFlags flags = GetCommFlags(domain);
   const MPI_Datatype baseType = MpiRealType();

   for (Index_t i = 0; i < 26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   if (flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp();
      const int recvCount = dx * dy * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }
   if (flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp();
      const int recvCount = dx * dy * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }
   if (flags.rowMin && doRecv) {
      const int fromRank = myRank - domain.tp();
      const int recvCount = dx * dz * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }
   if (flags.rowMax) {
      const int fromRank = myRank + domain.tp();
      const int recvCount = dx * dz * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }
   if (flags.colMin && doRecv) {
      const int fromRank = myRank - 1;
      const int recvCount = dy * dz * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }
   if (flags.colMax) {
      const int fromRank = myRank + 1;
      const int recvCount = dy * dz * xferFields;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm,
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
      ++pmsg;
   }

   if (planeOnly) {
      return;
   }

   if (flags.rowMin && flags.colMin && doRecv) {
      const int fromRank = myRank - domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dz * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMin && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() - domain.tp();
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dx * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.colMin && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dy * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMax && flags.colMax) {
      const int fromRank = myRank + domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dz * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMax && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() + domain.tp();
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dx * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.colMax && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dy * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMax && flags.colMin) {
      const int fromRank = myRank + domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dz * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMin && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() - domain.tp();
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dx * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.colMin && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dy * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMin && flags.colMax && doRecv) {
      const int fromRank = myRank - domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dz * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.rowMax && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() + domain.tp();
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dx * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }
   if (flags.colMax && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm,
                dy * xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg]);
      ++emsg;
   }

   if (flags.rowMin && flags.colMin && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMin && flags.colMin && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMin && doRecv) {
      const int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMax) {
      const int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1;
      MPI_Irecv(domain.commDataRecv.data() + pmsg * maxPlaneComm +
                   emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL,
                xferFields, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg + emsg + cmsg]);
      ++cmsg;
   }
}

void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly)
{
   if (domain.numRanks() == 1) {
      return;
   }

   const Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   const Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize();
   Index_t pmsg = 0;
   Index_t emsg = 0;
   Index_t cmsg = 0;
   int myRank = 0;
   const CommFlags flags = GetCommFlags(domain);
   const MPI_Datatype baseType = MpiRealType();
   std::array<MPI_Status, 26> status{};
   auto fields = CreateHostFieldCopies(domain, xferFields, fieldData);

   for (Index_t i = 0; i < 26; ++i) {
      domain.sendRequest[i] = MPI_REQUEST_NULL;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   if (flags.planeMin || flags.planeMax) {
      const Index_t sendCount = dx * dy;
      if (flags.planeMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < sendCount; ++i) {
               destAddr[i] = fields[fi](i);
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank - domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
      if (flags.planeMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < sendCount; ++i) {
               destAddr[i] = fields[fi](dx*dy*(dz - 1) + i);
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank + domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
   }

   if (flags.rowMin || flags.rowMax) {
      const Index_t sendCount = dx * dz;
      if (flags.rowMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  destAddr[i*dx + j] = fields[fi](i*dx*dy + j);
               }
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
      if (flags.rowMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  destAddr[i*dx + j] = fields[fi](dx*(dy - 1) + i*dx*dy + j);
               }
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
   }

   if (flags.colMin || flags.colMax) {
      const Index_t sendCount = dy * dz;
      if (flags.colMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  destAddr[i*dy + j] = fields[fi](i*dx*dy + j*dx);
               }
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
      if (flags.colMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  destAddr[i*dy + j] = fields[fi](dx - 1 + i*dx*dy + j*dx);
               }
            }
            destAddr += sendCount;
         }
         destAddr -= xferFields * sendCount;
         MPI_Isend(destAddr, xferFields * sendCount, baseType,
                   myRank + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]);
         ++pmsg;
      }
   }

   if (!planeOnly) {
      if (flags.rowMin && flags.colMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               destAddr[i] = fields[fi](i*dx*dy);
            }
            destAddr += dz;
         }
         destAddr -= xferFields * dz;
         MPI_Isend(destAddr, xferFields * dz, baseType, myRank - domain.tp() - 1,
                   msgType, MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMin && flags.planeMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dx; ++i) {
               destAddr[i] = fields[fi](i);
            }
            destAddr += dx;
         }
         destAddr -= xferFields * dx;
         MPI_Isend(destAddr, xferFields * dx, baseType,
                   myRank - domain.tp()*domain.tp() - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.colMin && flags.planeMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dy; ++i) {
               destAddr[i] = fields[fi](i*dx);
            }
            destAddr += dy;
         }
         destAddr -= xferFields * dy;
         MPI_Isend(destAddr, xferFields * dy, baseType,
                   myRank - domain.tp()*domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMax && flags.colMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               destAddr[i] = fields[fi](dx*dy - 1 + i*dx*dy);
            }
            destAddr += dz;
         }
         destAddr -= xferFields * dz;
         MPI_Isend(destAddr, xferFields * dz, baseType,
                   myRank + domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMax && flags.planeMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dx; ++i) {
               destAddr[i] = fields[fi](dx*(dy - 1) + dx*dy*(dz - 1) + i);
            }
            destAddr += dx;
         }
         destAddr -= xferFields * dx;
         MPI_Isend(destAddr, xferFields * dx, baseType,
                   myRank + domain.tp()*domain.tp() + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.colMax && flags.planeMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dy; ++i) {
               destAddr[i] = fields[fi](dx*dy*(dz - 1) + dx - 1 + i*dx);
            }
            destAddr += dy;
         }
         destAddr -= xferFields * dy;
         MPI_Isend(destAddr, xferFields * dy, baseType,
                   myRank + domain.tp()*domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMax && flags.colMin && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               destAddr[i] = fields[fi](dx*(dy - 1) + i*dx*dy);
            }
            destAddr += dz;
         }
         destAddr -= xferFields * dz;
         MPI_Isend(destAddr, xferFields * dz, baseType,
                   myRank + domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMin && flags.planeMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dx; ++i) {
               destAddr[i] = fields[fi](dx*dy*(dz - 1) + i);
            }
            destAddr += dx;
         }
         destAddr -= xferFields * dx;
         MPI_Isend(destAddr, xferFields * dx, baseType,
                   myRank + domain.tp()*domain.tp() - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.colMin && flags.planeMax && doSend) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dy; ++i) {
               destAddr[i] = fields[fi](dx*dy*(dz - 1) + i*dx);
            }
            destAddr += dy;
         }
         destAddr -= xferFields * dy;
         MPI_Isend(destAddr, xferFields * dy, baseType,
                   myRank + domain.tp()*domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMin && flags.colMax) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               destAddr[i] = fields[fi](dx - 1 + i*dx*dy);
            }
            destAddr += dz;
         }
         destAddr -= xferFields * dz;
         MPI_Isend(destAddr, xferFields * dz, baseType,
                   myRank - domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMax && flags.planeMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dx; ++i) {
               destAddr[i] = fields[fi](dx*(dy - 1) + i);
            }
            destAddr += dx;
         }
         destAddr -= xferFields * dx;
         MPI_Isend(destAddr, xferFields * dx, baseType,
                   myRank - domain.tp()*domain.tp() + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.colMax && flags.planeMin) {
         Real_t *destAddr = domain.commDataSend.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dy; ++i) {
               destAddr[i] = fields[fi](dx - 1 + i*dx);
            }
            destAddr += dy;
         }
         destAddr -= xferFields * dy;
         MPI_Isend(destAddr, xferFields * dy, baseType,
                   myRank - domain.tp()*domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg]);
         ++emsg;
      }

      if (flags.rowMin && flags.colMin && flags.planeMin) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](0);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank - domain.tp()*domain.tp() - domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMin && flags.colMin && flags.planeMax && doSend) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*dy*(dz - 1);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank + domain.tp()*domain.tp() - domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMin && flags.colMax && flags.planeMin) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx - 1;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank - domain.tp()*domain.tp() - domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMin && flags.colMax && flags.planeMax && doSend) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*dy*(dz - 1) + (dx - 1);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank + domain.tp()*domain.tp() - domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMax && flags.colMin && flags.planeMin) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*(dy - 1);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank - domain.tp()*domain.tp() + domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMax && flags.colMin && flags.planeMax && doSend) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank + domain.tp()*domain.tp() + domain.tp() - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMax && flags.colMax && flags.planeMin) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*dy - 1;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank - domain.tp()*domain.tp() + domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
      if (flags.rowMax && flags.colMax && flags.planeMax && doSend) {
         Real_t *comBuf = domain.commDataSend.data() + pmsg * maxPlaneComm +
                          emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
         const Index_t idx = dx*dy*dz - 1;
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            comBuf[fi] = fields[fi](idx);
         }
         MPI_Isend(comBuf, xferFields, baseType,
                   myRank + domain.tp()*domain.tp() + domain.tp() + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg + emsg + cmsg]);
         ++cmsg;
      }
   }

   MPI_Waitall(26, domain.sendRequest.data(), status.data());
}

void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData)
{
   if (domain.numRanks() == 1) {
      return;
   }

   const Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   const Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize();
   Index_t pmsg = 0;
   Index_t emsg = 0;
   Index_t cmsg = 0;
   const Index_t dx = domain.sizeX() + 1;
   const Index_t dy = domain.sizeY() + 1;
   const Index_t dz = domain.sizeZ() + 1;
   MPI_Status status{};
   const CommFlags flags = GetCommFlags(domain);
   auto fields = CreateHostFieldCopies(domain, xferFields, fieldData);

   if (flags.planeMin || flags.planeMax) {
      const Index_t opCount = dx * dy;
      if (flags.planeMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](i) += srcAddr[i];
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.planeMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](dx*dy*(dz - 1) + i) += srcAddr[i];
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.rowMin || flags.rowMax) {
      const Index_t opCount = dx * dz;
      if (flags.rowMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  fields[fi](i*dx*dy + j) += srcAddr[i*dx + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.rowMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  fields[fi](dx*(dy - 1) + i*dx*dy + j) += srcAddr[i*dx + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.colMin || flags.colMax) {
      const Index_t opCount = dy * dz;
      if (flags.colMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  fields[fi](i*dx*dy + j*dx) += srcAddr[i*dy + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.colMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  fields[fi](dx - 1 + i*dx*dy + j*dx) += srcAddr[i*dy + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.rowMin && flags.colMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](i*dx*dy) += srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.planeMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](i) += srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMin && flags.planeMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](i*dx) += srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.colMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx*dy - 1 + i*dx*dy) += srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*(dy - 1) + dx*dy*(dz - 1) + i) += srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMax && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx*dy*(dz - 1) + dx - 1 + i*dx) += srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.colMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx*(dy - 1) + i*dx*dy) += srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*dy*(dz - 1) + i) += srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMin && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx*dy*(dz - 1) + i*dx) += srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.colMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx - 1 + i*dx*dy) += srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.planeMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*(dy - 1) + i) += srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMax && flags.planeMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx - 1 + i*dx) += srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }

   if (flags.rowMin && flags.colMin && flags.planeMin) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](0) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMin && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMin) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1) + (dx - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMin) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*(dy - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMin) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*dz - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) += comBuf[fi];
      }
      ++cmsg;
   }

   CopyFieldsBack(domain, xferFields, fieldData, fields);
}

void CommSyncPosVel(Domain& domain)
{
   if (domain.numRanks() == 1) {
      return;
   }

   const bool doRecv = false;
   const Index_t xferFields = 6;
   Domain_member fieldData[6] = {
      &Domain::x, &Domain::y, &Domain::z,
      &Domain::xd, &Domain::yd, &Domain::zd
   };
   const Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   const Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize();
   Index_t pmsg = 0;
   Index_t emsg = 0;
   Index_t cmsg = 0;
   const Index_t dx = domain.sizeX() + 1;
   const Index_t dy = domain.sizeY() + 1;
   const Index_t dz = domain.sizeZ() + 1;
   MPI_Status status{};
   const CommFlags flags = GetCommFlags(domain);
   auto fields = CreateHostFieldCopies(domain, xferFields, fieldData);

   if (flags.planeMin || flags.planeMax) {
      const Index_t opCount = dx * dy;
      if (flags.planeMin && doRecv) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](i) = srcAddr[i];
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.planeMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](dx*dy*(dz - 1) + i) = srcAddr[i];
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.rowMin || flags.rowMax) {
      const Index_t opCount = dx * dz;
      if (flags.rowMin && doRecv) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  fields[fi](i*dx*dy + j) = srcAddr[i*dx + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.rowMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dx; ++j) {
                  fields[fi](dx*(dy - 1) + i*dx*dy + j) = srcAddr[i*dx + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.colMin || flags.colMax) {
      const Index_t opCount = dy * dz;
      if (flags.colMin && doRecv) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  fields[fi](i*dx*dy + j*dx) = srcAddr[i*dy + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
      if (flags.colMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < dz; ++i) {
               for (Index_t j = 0; j < dy; ++j) {
                  fields[fi](dx - 1 + i*dx*dy + j*dx) = srcAddr[i*dy + j];
               }
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.rowMin && flags.colMin && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](i*dx*dy) = srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.planeMin && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](i) = srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMin && flags.planeMin && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](i*dx) = srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.colMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx*dy - 1 + i*dx*dy) = srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*(dy - 1) + dx*dy*(dz - 1) + i) = srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMax && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx*dy*(dz - 1) + dx - 1 + i*dx) = srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.colMin) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx*(dy - 1) + i*dx*dy) = srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*dy*(dz - 1) + i) = srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMin && flags.planeMax) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx*dy*(dz - 1) + i*dx) = srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }
   if (flags.rowMin && flags.colMax && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dz; ++i) {
            fields[fi](dx - 1 + i*dx*dy) = srcAddr[i];
         }
         srcAddr += dz;
      }
      ++emsg;
   }
   if (flags.rowMax && flags.planeMin && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dx; ++i) {
            fields[fi](dx*(dy - 1) + i) = srcAddr[i];
         }
         srcAddr += dx;
      }
      ++emsg;
   }
   if (flags.colMax && flags.planeMin && doRecv) {
      Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm + emsg * maxEdgeComm;
      MPI_Wait(&domain.recvRequest[pmsg + emsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         for (Index_t i = 0; i < dy; ++i) {
            fields[fi](dx - 1 + i*dx) = srcAddr[i];
         }
         srcAddr += dy;
      }
      ++emsg;
   }

   if (flags.rowMin && flags.colMin && flags.planeMin && doRecv) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](0) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMin && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMin && doRecv) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMin && flags.colMax && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1) + (dx - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMin && doRecv) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*(dy - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMin && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1);
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMin && doRecv) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }
   if (flags.rowMax && flags.colMax && flags.planeMax) {
      Real_t *comBuf = domain.commDataRecv.data() + pmsg * maxPlaneComm +
                       emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
      const Index_t idx = dx*dy*dz - 1;
      MPI_Wait(&domain.recvRequest[pmsg + emsg + cmsg], &status);
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         fields[fi](idx) = comBuf[fi];
      }
      ++cmsg;
   }

   CopyFieldsBack(domain, xferFields, fieldData, fields);
}

void CommMonoQ(Domain& domain)
{
   if (domain.numRanks() == 1) {
      return;
   }

   const Index_t xferFields = 3;
   Domain_member fieldData[3] = {
      &Domain::delv_xi,
      &Domain::delv_eta,
      &Domain::delv_zeta
   };
   std::array<Index_t, 3> fieldOffset = {
      domain.numElem(),
      domain.numElem(),
      domain.numElem()
   };
   const Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   Index_t pmsg = 0;
   const CommFlags flags = GetCommFlags(domain);
   MPI_Status status{};
   auto fields = CreateHostFieldCopies(domain, xferFields, fieldData);

   const Index_t dx = domain.sizeX();
   const Index_t dy = domain.sizeY();
   const Index_t dz = domain.sizeZ();

   if (flags.planeMin || flags.planeMax) {
      const Index_t opCount = dx * dy;
      if (flags.planeMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
            fieldOffset[fi] += opCount;
         }
         ++pmsg;
      }
      if (flags.planeMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
            fieldOffset[fi] += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.rowMin || flags.rowMax) {
      const Index_t opCount = dx * dz;
      if (flags.rowMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
            fieldOffset[fi] += opCount;
         }
         ++pmsg;
      }
      if (flags.rowMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
            fieldOffset[fi] += opCount;
         }
         ++pmsg;
      }
   }

   if (flags.colMin || flags.colMax) {
      const Index_t opCount = dy * dz;
      if (flags.colMin) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
            fieldOffset[fi] += opCount;
         }
         ++pmsg;
      }
      if (flags.colMax) {
         Real_t *srcAddr = domain.commDataRecv.data() + pmsg * maxPlaneComm;
         MPI_Wait(&domain.recvRequest[pmsg], &status);
         for (Index_t fi = 0; fi < xferFields; ++fi) {
            for (Index_t i = 0; i < opCount; ++i) {
               fields[fi](fieldOffset[fi] + i) = srcAddr[i];
            }
            srcAddr += opCount;
         }
         ++pmsg;
      }
   }

   CopyFieldsBack(domain, xferFields, fieldData, fields);
}

#else

void CommRecv(Domain&, Int_t, Index_t, Index_t, Index_t, Index_t, bool, bool) {}

void CommSend(Domain&, Int_t, Index_t, Domain_member*, Index_t, Index_t, Index_t,
              bool, bool) {}

void CommSBN(Domain&, Int_t, Domain_member*) {}

void CommSyncPosVel(Domain&) {}

void CommMonoQ(Domain&) {}

#endif
