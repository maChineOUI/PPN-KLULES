#include "lulesh-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <optional>

#if USE_MPI
#include <mpi.h>

namespace {

struct RuntimeState {
   int rank = 0 ;
   int size = 1 ;
   bool initialized = false ;
   std::optional<DistributedComm> world ;
} ;

auto runtime() -> RuntimeState&
{
   static RuntimeState state ;
   return state ;
}

[[noreturn]] auto RuntimeAbort(const char* message) -> void
{
   std::fprintf(stderr, "LULESH runtime error: %s\n", message) ;
   std::abort() ;
}

} // namespace

void InitDistributedRuntime(int& argc, char**& argv)
{
#ifdef _OPENMP
   int thread_support = 0 ;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support) ;
#else
   MPI_Init(&argc, &argv) ;
#endif

   auto& state = runtime() ;
   state.initialized = true ;
   MPI_Comm_rank(MPI_COMM_WORLD, &state.rank) ;
   MPI_Comm_size(MPI_COMM_WORLD, &state.size) ;
}

void BindWorldCommunicator(const Kokkos::DefaultExecutionSpace& exec)
{
   auto& state = runtime() ;
   if (!state.initialized) {
      RuntimeAbort("BindWorldCommunicator called before InitDistributedRuntime") ;
   }

   state.world = DistributedComm::from_raw(MPI_COMM_WORLD, exec) ;
}

void FinalizeDistributedRuntime()
{
   auto& state = runtime() ;
   state.world.reset() ;
   if (state.initialized) {
      MPI_Finalize() ;
      state.initialized = false ;
   }
}

int DistributedRank()
{
   return runtime().rank ;
}

int DistributedSize()
{
   return runtime().size ;
}

auto WorldComm() -> DistributedComm&
{
   auto& state = runtime() ;
   if (!state.world) {
      RuntimeAbort("World communicator requested before BindWorldCommunicator") ;
   }
   return *state.world ;
}

#else

void InitDistributedRuntime(int& argc, char**& argv)
{
   (void)argc ;
   (void)argv ;
}

void BindWorldCommunicator(const Kokkos::DefaultExecutionSpace& exec)
{
   (void)exec ;
}

void FinalizeDistributedRuntime() {}

int DistributedRank()
{
   return 0 ;
}

int DistributedSize()
{
   return 1 ;
}

#endif
