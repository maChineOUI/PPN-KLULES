#include "lulesh-profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "lulesh-runtime.h"

namespace {

using HostView = Kokkos::View<double*, Kokkos::HostSpace> ;
using ProfileClock = std::chrono::steady_clock ;

struct ProfileState {
   std::array<double, kProfileTimerCount> totals {} ;
   std::array<double, kProfileTimerCount> counts {} ;
   std::array<double, kProfileFaceCount> monoqRecvWaitFaceTotals {} ;
   std::array<double, kProfileFaceCount> monoqRecvWaitFaceCounts {} ;
   std::array<double, kProfileFaceCount> monoqSendWaitFaceTotals {} ;
   std::array<double, kProfileFaceCount> monoqSendWaitFaceCounts {} ;
   std::array<double, kProfileFaceCount> monoqRecvLifeFaceTotals {} ;
   std::array<double, kProfileFaceCount> monoqRecvLifeFaceCounts {} ;
   std::array<double, kProfileFaceCount> monoqSendLifeFaceTotals {} ;
   std::array<double, kProfileFaceCount> monoqSendLifeFaceCounts {} ;
   std::array<ProfileClock::time_point, kProfileFaceCount> monoqRecvPostStamp {} ;
   std::array<ProfileClock::time_point, kProfileFaceCount> monoqSendPostStamp {} ;
   std::array<bool, kProfileFaceCount> monoqRecvPostArmed {} ;
   std::array<bool, kProfileFaceCount> monoqSendPostArmed {} ;
} ;

auto state() -> ProfileState&
{
   static ProfileState s ;
   return s ;
}

constexpr std::array<const char*, kProfileTimerCount> kTimerNames = {
   "step_total",
   "dt_allreduce",
   "sbn_host_pull",
   "sbn_post",
   "sbn_wait_unpack",
   "sbn_host_push",
   "monoq_host_pull",
   "monoq_post",
   "monoq_recv_post",
   "monoq_send_post",
   "monoq_interior",
   "monoq_rowcol_wait_unpack",
   "monoq_rowcol_boundary",
   "monoq_wait_unpack",
   "monoq_boundary",
   "commrecv_post",
   "commsend_pack",
   "commsend_post",
   "commsbn_wait",
   "commsbn_unpack_add",
   "commmonoq_wait",
   "commmonoq_unpack",
   "commmonoq_host_push",
   "commmonoq_send_wait"
} ;

constexpr std::array<const char*, kProfileFaceCount> kFaceNames = {
   "plane_min",
   "plane_max",
   "row_min",
   "row_max",
   "col_min",
   "col_max"
} ;

template <std::size_t N>
auto FillView(HostView view, const std::array<double, N>& values) -> void
{
   for (std::size_t i = 0; i < N; ++i) {
      view(i) = values[i] ;
   }
}

template <std::size_t N>
auto ReadView(const HostView& view) -> std::array<double, N>
{
   std::array<double, N> values {} ;
   for (std::size_t i = 0; i < N; ++i) {
      values[i] = view(i) ;
   }
   return values ;
}

#if USE_MPI
template <std::size_t N, typename Op>
auto AllreduceArray(const std::array<double, N>& local,
                    Op op,
                    const char* sendName,
                    const char* recvName) -> std::array<double, N>
{
   HostView send(sendName, N) ;
   HostView recv(recvName, N) ;
   FillView(send, local) ;
   KokkosComm::Experimental::allreduce(WorldComm(), send, recv, op).wait() ;
   return ReadView<N>(recv) ;
}
#endif

} // namespace

auto ProfileReset() -> void
{
   auto& s = state() ;
   s.totals.fill(0.0) ;
   s.counts.fill(0.0) ;
   s.monoqRecvWaitFaceTotals.fill(0.0) ;
   s.monoqRecvWaitFaceCounts.fill(0.0) ;
   s.monoqSendWaitFaceTotals.fill(0.0) ;
   s.monoqSendWaitFaceCounts.fill(0.0) ;
   s.monoqRecvLifeFaceTotals.fill(0.0) ;
   s.monoqRecvLifeFaceCounts.fill(0.0) ;
   s.monoqSendLifeFaceTotals.fill(0.0) ;
   s.monoqSendLifeFaceCounts.fill(0.0) ;
   s.monoqRecvPostArmed.fill(false) ;
   s.monoqSendPostArmed.fill(false) ;
}

auto ProfileAddSample(ProfileTimer timer, double seconds) -> void
{
   if (!kLuleshProfileEnabled) {
      return ;
   }

   auto idx = static_cast<std::size_t>(timer) ;
   auto& s = state() ;
   s.totals[idx] += seconds ;
   s.counts[idx] += 1.0 ;
}

auto ProfileAddMonoqRecvWaitFaceSample(std::size_t face, double seconds) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   s.monoqRecvWaitFaceTotals[face] += seconds ;
   s.monoqRecvWaitFaceCounts[face] += 1.0 ;
}

auto ProfileAddMonoqSendWaitFaceSample(std::size_t face, double seconds) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   s.monoqSendWaitFaceTotals[face] += seconds ;
   s.monoqSendWaitFaceCounts[face] += 1.0 ;
}

auto ProfileMarkMonoqRecvPost(std::size_t face) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   s.monoqRecvPostStamp[face] = ProfileClock::now() ;
   s.monoqRecvPostArmed[face] = true ;
}

auto ProfileMarkMonoqSendPost(std::size_t face) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   s.monoqSendPostStamp[face] = ProfileClock::now() ;
   s.monoqSendPostArmed[face] = true ;
}

auto ProfileCompleteMonoqRecvFace(std::size_t face) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   if (!s.monoqRecvPostArmed[face]) {
      return ;
   }

   auto elapsed =
      std::chrono::duration<double>(ProfileClock::now() - s.monoqRecvPostStamp[face]) ;
   s.monoqRecvLifeFaceTotals[face] += elapsed.count() ;
   s.monoqRecvLifeFaceCounts[face] += 1.0 ;
   s.monoqRecvPostArmed[face] = false ;
}

auto ProfileCompleteMonoqSendFace(std::size_t face) -> void
{
   if (!kLuleshProfileEnabled || face >= kProfileFaceCount) {
      return ;
   }

   auto& s = state() ;
   if (!s.monoqSendPostArmed[face]) {
      return ;
   }

   auto elapsed =
      std::chrono::duration<double>(ProfileClock::now() - s.monoqSendPostStamp[face]) ;
   s.monoqSendLifeFaceTotals[face] += elapsed.count() ;
   s.monoqSendLifeFaceCounts[face] += 1.0 ;
   s.monoqSendPostArmed[face] = false ;
}

auto ProfileFaceName(std::size_t face) -> const char*
{
   return face < kProfileFaceCount ? kFaceNames[face] : "unknown_face" ;
}

auto ProfileTimerName(ProfileTimer timer) -> const char*
{
   return kTimerNames[static_cast<std::size_t>(timer)] ;
}

auto ProfileDumpSummary(int cycles, int numRanks, int myRank) -> void
{
   if (!kLuleshProfileEnabled) {
      return ;
   }

   auto& s = state() ;

   auto sumTotals = s.totals ;
   auto maxTotals = s.totals ;
   auto minTotals = s.totals ;
   auto sumCounts = s.counts ;
   auto monoqFaceSumTotals = s.monoqRecvWaitFaceTotals ;
   auto monoqFaceMaxTotals = s.monoqRecvWaitFaceTotals ;
   auto monoqFaceMinTotals = s.monoqRecvWaitFaceTotals ;
   auto monoqFaceSumCounts = s.monoqRecvWaitFaceCounts ;
   std::array<double, kProfileFaceCount> monoqFaceActiveRanks {} ;
   auto monoqSendFaceSumTotals = s.monoqSendWaitFaceTotals ;
   auto monoqSendFaceSumCounts = s.monoqSendWaitFaceCounts ;
   auto monoqRecvLifeFaceSumTotals = s.monoqRecvLifeFaceTotals ;
   auto monoqRecvLifeFaceSumCounts = s.monoqRecvLifeFaceCounts ;
   auto monoqSendLifeFaceSumTotals = s.monoqSendLifeFaceTotals ;
   auto monoqSendLifeFaceSumCounts = s.monoqSendLifeFaceCounts ;

   for (std::size_t i = 0; i < kProfileTimerCount; ++i) {
      if (sumCounts[i] == 0.0) {
         minTotals[i] = std::numeric_limits<double>::infinity() ;
      }
   }

   for (std::size_t face = 0; face < kProfileFaceCount; ++face) {
      if (monoqFaceSumCounts[face] == 0.0) {
         monoqFaceMinTotals[face] = std::numeric_limits<double>::infinity() ;
         continue ;
      }
      monoqFaceActiveRanks[face] = 1.0 ;
   }

#if USE_MPI
   if (numRanks > 1) {
      sumTotals = AllreduceArray(s.totals, KokkosComm::Sum{},
                                 "profile_sum_send", "profile_sum_recv") ;
      maxTotals = AllreduceArray(s.totals, KokkosComm::Max{},
                                 "profile_max_send", "profile_max_recv") ;
      minTotals = AllreduceArray(minTotals, KokkosComm::Min{},
                                 "profile_min_send", "profile_min_recv") ;
      sumCounts = AllreduceArray(s.counts, KokkosComm::Sum{},
                                 "profile_count_send", "profile_count_recv") ;
      monoqFaceSumTotals =
         AllreduceArray(monoqFaceSumTotals, KokkosComm::Sum{},
                        "profile_monoq_face_sum_send", "profile_monoq_face_sum_recv") ;
      monoqFaceMaxTotals =
         AllreduceArray(monoqFaceMaxTotals, KokkosComm::Max{},
                        "profile_monoq_face_max_send", "profile_monoq_face_max_recv") ;
      monoqFaceMinTotals =
         AllreduceArray(monoqFaceMinTotals, KokkosComm::Min{},
                        "profile_monoq_face_min_send", "profile_monoq_face_min_recv") ;
      monoqFaceSumCounts =
         AllreduceArray(monoqFaceSumCounts, KokkosComm::Sum{},
                        "profile_monoq_face_count_send", "profile_monoq_face_count_recv") ;
      monoqFaceActiveRanks =
         AllreduceArray(monoqFaceActiveRanks, KokkosComm::Sum{},
                        "profile_monoq_face_active_send", "profile_monoq_face_active_recv") ;
      monoqSendFaceSumTotals =
         AllreduceArray(monoqSendFaceSumTotals, KokkosComm::Sum{},
                        "profile_monoq_send_face_sum_send",
                        "profile_monoq_send_face_sum_recv") ;
      monoqSendFaceSumCounts =
         AllreduceArray(monoqSendFaceSumCounts, KokkosComm::Sum{},
                        "profile_monoq_send_face_count_send",
                        "profile_monoq_send_face_count_recv") ;
      monoqRecvLifeFaceSumTotals =
         AllreduceArray(monoqRecvLifeFaceSumTotals, KokkosComm::Sum{},
                        "profile_monoq_recv_life_sum_send",
                        "profile_monoq_recv_life_sum_recv") ;
      monoqRecvLifeFaceSumCounts =
         AllreduceArray(monoqRecvLifeFaceSumCounts, KokkosComm::Sum{},
                        "profile_monoq_recv_life_count_send",
                        "profile_monoq_recv_life_count_recv") ;
      monoqSendLifeFaceSumTotals =
         AllreduceArray(monoqSendLifeFaceSumTotals, KokkosComm::Sum{},
                        "profile_monoq_send_life_sum_send",
                        "profile_monoq_send_life_sum_recv") ;
      monoqSendLifeFaceSumCounts =
         AllreduceArray(monoqSendLifeFaceSumCounts, KokkosComm::Sum{},
                        "profile_monoq_send_life_count_send",
                        "profile_monoq_send_life_count_recv") ;
   }
#else
   (void)numRanks ;
#endif

   if (myRank != 0) {
      return ;
   }

   auto stepAvgTotal = 0.0 ;
   auto stepIdx = static_cast<std::size_t>(ProfileTimer::step_total) ;
   if (sumCounts[stepIdx] > 0.0 && numRanks > 0) {
      stepAvgTotal = sumTotals[stepIdx] / static_cast<double>(numRanks) ;
   }

   auto avgTotal = [&](ProfileTimer timer) -> double {
      auto idx = static_cast<std::size_t>(timer) ;
      return sumTotals[idx] / static_cast<double>(numRanks) ;
   } ;

   auto perStepMs = [&](ProfileTimer timer) -> double {
      if (cycles <= 0) {
         return 0.0 ;
      }
      return 1000.0 * avgTotal(timer) / static_cast<double>(cycles) ;
   } ;

   std::printf("[profile] summary_begin cycles=%d ranks=%d\n", cycles, numRanks) ;
   for (std::size_t i = 0; i < kProfileTimerCount; ++i) {
      if (sumCounts[i] == 0.0) {
         continue ;
      }

      auto avgTotal = sumTotals[i] / static_cast<double>(numRanks) ;
      auto avgMs = 1000.0 * avgTotal ;
      auto maxMs = 1000.0 * maxTotals[i] ;
      auto minMs = std::isfinite(minTotals[i]) ? 1000.0 * minTotals[i] : 0.0 ;
      auto avgCalls = sumCounts[i] / static_cast<double>(numRanks) ;
      auto perStepMs = cycles > 0 ? avgMs / static_cast<double>(cycles) : 0.0 ;
      auto share = stepAvgTotal > 0.0 ? avgTotal / stepAvgTotal : 0.0 ;
      auto imbalance = avgTotal > 0.0 ? maxTotals[i] / avgTotal : 0.0 ;

      std::printf(
         "[profile] timer=%s avg_ms=%.6f max_ms=%.6f min_ms=%.6f "
         "avg_calls=%.2f per_step_ms=%.6f share=%.6f imbalance=%.6f\n",
         kTimerNames[i], avgMs, maxMs, minMs, avgCalls, perStepMs, share, imbalance) ;
   }

   auto monoqWindowMs = perStepMs(ProfileTimer::monoq_interior) ;
   auto monoqRowColWaitUnpackMs = perStepMs(ProfileTimer::monoq_rowcol_wait_unpack) ;
   auto monoqRowColBoundaryMs = perStepMs(ProfileTimer::monoq_rowcol_boundary) ;
   auto monoqWaitUnpackMs = perStepMs(ProfileTimer::monoq_wait_unpack) ;
   auto monoqRecvWaitMs = perStepMs(ProfileTimer::commmonoq_wait) ;
   auto monoqSendWaitMs = perStepMs(ProfileTimer::commmonoq_send_wait) ;
   auto monoqUnpackMs = perStepMs(ProfileTimer::commmonoq_unpack) ;
   auto monoqHostPushMs = perStepMs(ProfileTimer::commmonoq_host_push) ;
   auto monoqResidualSyncMs = monoqRecvWaitMs + monoqSendWaitMs ;
   auto monoqTailAfterWindowMs =
      monoqRecvWaitMs + monoqSendWaitMs + monoqUnpackMs + monoqHostPushMs ;
   auto monoqPrePlaneComputeMs = monoqWindowMs + monoqRowColBoundaryMs ;
   auto monoqPrePlaneActivityMs =
      monoqWindowMs + monoqRowColWaitUnpackMs + monoqRowColBoundaryMs ;

   if (sumCounts[static_cast<std::size_t>(ProfileTimer::monoq_post)] > 0.0 && cycles > 0) {
      auto monoqPostMs = perStepMs(ProfileTimer::monoq_post) ;
      auto monoqRecvPostMs = perStepMs(ProfileTimer::monoq_recv_post) ;
      auto monoqSendPostMs = perStepMs(ProfileTimer::monoq_send_post) ;
      auto postToWindow = monoqWindowMs > 0.0 ? monoqPostMs / monoqWindowMs : 0.0 ;
      auto residualSyncToWindow =
         monoqWindowMs > 0.0 ? monoqResidualSyncMs / monoqWindowMs : 0.0 ;
      auto tailToWindow =
         monoqWindowMs > 0.0 ? monoqTailAfterWindowMs / monoqWindowMs : 0.0 ;

      std::printf(
         "[profile] monoq_overlap post_ms=%.6f recv_post_ms=%.6f send_post_ms=%.6f "
         "window_ms=%.6f wait_unpack_ms=%.6f recv_wait_ms=%.6f send_wait_ms=%.6f "
         "unpack_ms=%.6f host_push_ms=%.6f residual_sync_ms=%.6f "
         "post_to_window=%.6f residual_sync_to_window=%.6f tail_to_window=%.6f\n",
         monoqPostMs, monoqRecvPostMs, monoqSendPostMs,
         monoqWindowMs, monoqWaitUnpackMs, monoqRecvWaitMs, monoqSendWaitMs,
         monoqUnpackMs, monoqHostPushMs, monoqResidualSyncMs,
         postToWindow, residualSyncToWindow, tailToWindow) ;

      std::printf(
         "[profile] monoq_overlap_staged rowcol_wait_unpack_ms=%.6f "
         "rowcol_boundary_ms=%.6f pre_plane_compute_ms=%.6f "
         "pre_plane_activity_ms=%.6f plane_wait_unpack_ms=%.6f\n",
         monoqRowColWaitUnpackMs, monoqRowColBoundaryMs,
         monoqPrePlaneComputeMs, monoqPrePlaneActivityMs, monoqWaitUnpackMs) ;
   }

   auto dominantFace = static_cast<std::size_t>(0) ;
   auto dominantFacePerStepMs = 0.0 ;
   for (std::size_t face = 0; face < kProfileFaceCount; ++face) {
      auto activeRanks = monoqFaceActiveRanks[face] ;
      if (activeRanks == 0.0) {
         continue ;
      }

      auto avgTotal = monoqFaceSumTotals[face] / activeRanks ;
      auto avgMs = 1000.0 * avgTotal ;
      auto maxMs = 1000.0 * monoqFaceMaxTotals[face] ;
      auto minMs =
         std::isfinite(monoqFaceMinTotals[face]) ? 1000.0 * monoqFaceMinTotals[face] : 0.0 ;
      auto avgCalls = monoqFaceSumCounts[face] / activeRanks ;
      auto facePerStepMs = cycles > 0 ? avgMs / static_cast<double>(cycles) : 0.0 ;
      auto share = stepAvgTotal > 0.0 ? avgTotal / stepAvgTotal : 0.0 ;
      auto imbalance = avgTotal > 0.0 ? monoqFaceMaxTotals[face] / avgTotal : 0.0 ;

      if (facePerStepMs > dominantFacePerStepMs) {
         dominantFacePerStepMs = facePerStepMs ;
         dominantFace = face ;
      }

      std::printf(
         "[profile] monoq_face_wait face=%s active_ranks=%.0f avg_ms=%.6f "
         "max_ms=%.6f min_ms=%.6f avg_calls=%.2f per_step_ms=%.6f "
         "share=%.6f imbalance=%.6f\n",
         kFaceNames[face], activeRanks, avgMs, maxMs, minMs, avgCalls,
         facePerStepMs, share, imbalance) ;
   }

   if (cycles > 0 && monoqRecvWaitMs > 0.0) {
      auto dominantShareOfRecvWait = dominantFacePerStepMs / monoqRecvWaitMs ;
      std::printf(
         "[profile] monoq_face_wait_peak face=%s per_step_ms=%.6f "
         "share_of_recv_wait=%.6f\n",
         kFaceNames[dominantFace], dominantFacePerStepMs, dominantShareOfRecvWait) ;
   }

   for (std::size_t face = 0; face < 2; ++face) {
      auto activeRanks = monoqFaceActiveRanks[face] ;
      if (activeRanks == 0.0) {
         continue ;
      }

      auto recvWaitAvgTotal = monoqFaceSumTotals[face] / activeRanks ;
      auto recvLifeAvgTotal = monoqRecvLifeFaceSumCounts[face] > 0.0
         ? monoqRecvLifeFaceSumTotals[face] / activeRanks
         : 0.0 ;
      auto sendWaitAvgTotal = monoqSendFaceSumCounts[face] > 0.0
         ? monoqSendFaceSumTotals[face] / activeRanks
         : 0.0 ;
      auto sendLifeAvgTotal = monoqSendLifeFaceSumCounts[face] > 0.0
         ? monoqSendLifeFaceSumTotals[face] / activeRanks
         : 0.0 ;

      auto recvWaitPerStepMs =
         cycles > 0 ? 1000.0 * recvWaitAvgTotal / static_cast<double>(cycles) : 0.0 ;
      auto recvLifePerStepMs =
         cycles > 0 ? 1000.0 * recvLifeAvgTotal / static_cast<double>(cycles) : 0.0 ;
      auto sendWaitPerStepMs =
         cycles > 0 ? 1000.0 * sendWaitAvgTotal / static_cast<double>(cycles) : 0.0 ;
      auto sendLifePerStepMs =
         cycles > 0 ? 1000.0 * sendLifeAvgTotal / static_cast<double>(cycles) : 0.0 ;
      auto recvHiddenPerStepMs = std::max(0.0, recvLifePerStepMs - recvWaitPerStepMs) ;
      auto sendHiddenPerStepMs = std::max(0.0, sendLifePerStepMs - sendWaitPerStepMs) ;
      auto recvWaitFraction =
         recvLifePerStepMs > 0.0 ? recvWaitPerStepMs / recvLifePerStepMs : 0.0 ;
      auto sendWaitFraction =
         sendLifePerStepMs > 0.0 ? sendWaitPerStepMs / sendLifePerStepMs : 0.0 ;

      std::printf(
         "[profile] monoq_plane_lifecycle face=%s active_ranks=%.0f "
         "recv_life_ms=%.6f recv_wait_ms=%.6f recv_hidden_ms=%.6f "
         "recv_wait_fraction=%.6f send_life_ms=%.6f send_wait_ms=%.6f "
         "send_hidden_ms=%.6f send_wait_fraction=%.6f\n",
         kFaceNames[face], activeRanks,
         recvLifePerStepMs, recvWaitPerStepMs, recvHiddenPerStepMs, recvWaitFraction,
         sendLifePerStepMs, sendWaitPerStepMs, sendHiddenPerStepMs, sendWaitFraction) ;
   }

   std::printf("[profile] summary_end\n") ;
}
