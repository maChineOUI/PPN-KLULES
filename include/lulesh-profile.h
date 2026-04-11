#pragma once

#include <chrono>
#include <cstddef>

#include <Kokkos_Core.hpp>

#if defined(LULESH_ENABLE_PROFILE)
inline constexpr bool kLuleshProfileEnabled = true ;
#else
inline constexpr bool kLuleshProfileEnabled = false ;
#endif

enum class ProfileTimer : std::size_t {
   step_total = 0,
   dt_allreduce,
   sbn_host_pull,
   sbn_post,
   sbn_wait_unpack,
   sbn_host_push,
   monoq_host_pull,
   monoq_post,
   monoq_recv_post,
   monoq_send_post,
   monoq_interior,
   monoq_rowcol_wait_unpack,
   monoq_rowcol_boundary,
   monoq_wait_unpack,
   monoq_boundary,
   commrecv_post,
   commsend_pack,
   commsend_post,
   commsbn_wait,
   commsbn_unpack_add,
   commmonoq_wait,
   commmonoq_unpack,
   commmonoq_host_push,
   commmonoq_send_wait,
   count
} ;

inline constexpr auto kProfileTimerCount =
   static_cast<std::size_t>(ProfileTimer::count) ;
inline constexpr std::size_t kProfileFaceCount = 6 ;

auto ProfileReset() -> void ;
auto ProfileAddSample(ProfileTimer timer, double seconds) -> void ;
auto ProfileAddMonoqRecvWaitFaceSample(std::size_t face, double seconds) -> void ;
auto ProfileAddMonoqSendWaitFaceSample(std::size_t face, double seconds) -> void ;
auto ProfileMarkMonoqRecvPost(std::size_t face) -> void ;
auto ProfileMarkMonoqSendPost(std::size_t face) -> void ;
auto ProfileCompleteMonoqRecvFace(std::size_t face) -> void ;
auto ProfileCompleteMonoqSendFace(std::size_t face) -> void ;
auto ProfileFaceName(std::size_t face) -> const char* ;
auto ProfileTimerName(ProfileTimer timer) -> const char* ;
auto ProfileDumpSummary(int cycles, int numRanks, int myRank) -> void ;

class ProfileScope {
public:
   explicit ProfileScope(ProfileTimer timer, bool doFence = true)
      : m_timer(timer), m_doFence(doFence), m_active(kLuleshProfileEnabled)
   {
      if (!m_active) {
         return ;
      }
      if (m_doFence) {
         Kokkos::fence() ;
      }
      m_start = clock_type::now() ;
   }

   ~ProfileScope()
   {
      if (!m_active) {
         return ;
      }
      if (m_doFence) {
         Kokkos::fence() ;
      }
      auto elapsed = std::chrono::duration<double>(clock_type::now() - m_start) ;
      ProfileAddSample(m_timer, elapsed.count()) ;
   }

   ProfileScope(const ProfileScope&) = delete ;
   auto operator=(const ProfileScope&) -> ProfileScope& = delete ;

private:
   using clock_type = std::chrono::steady_clock ;

   ProfileTimer m_timer ;
   clock_type::time_point m_start {} ;
   bool m_doFence = true ;
   bool m_active = false ;
} ;
