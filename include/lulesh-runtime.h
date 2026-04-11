#pragma once

#include <Kokkos_Core.hpp>

#if USE_MPI
#include <KokkosComm/KokkosComm.hpp>

using DistributedComm =
   KokkosComm::Communicator<KokkosComm::MpiSpace, Kokkos::DefaultExecutionSpace> ;
#endif

void InitDistributedRuntime(int& argc, char**& argv);
void BindWorldCommunicator(const Kokkos::DefaultExecutionSpace& exec);
void FinalizeDistributedRuntime();

int DistributedRank();
int DistributedSize();

#if USE_MPI
auto WorldComm() -> DistributedComm&;
#endif
