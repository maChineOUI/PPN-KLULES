/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only, 
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt 
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.  
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <climits>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>

#include <chrono>

static inline double wall_clock()
{
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  const auto t1 = clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}


#if _OPENMP
# include <omp.h>
#endif

#include <Kokkos_Core.hpp>
#include "lulesh.h"


/* Work Routines
   Routines de travail
   工作例程
*/

static inline
void TimeIncrement(Domain& domain)
{
   Real_t targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain.deltatime() ;

      /* This will require a reduction in parallel
         Ceci nécessite une réduction en parallèle
         这一部分在并行实现中需要一次全局规约
      */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / Real_t(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

#if USE_MPI
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE
      Essayer d'éviter un très petit redimensionnement au prochain cycle
      尽量避免下一步时间步长出现非常小的缩放
   */
   if ((targetdt > domain.deltatime()) &&
       (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CollectDomainNodesToElemNodes(const Domain &domain,
                                          const Index_t* elemToNode,
                                          Real_t elemX[8],
                                          Real_t elemY[8],
                                          Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = domain.x(nd0i);
   elemX[1] = domain.x(nd1i);
   elemX[2] = domain.x(nd2i);
   elemX[3] = domain.x(nd3i);
   elemX[4] = domain.x(nd4i);
   elemX[5] = domain.x(nd5i);
   elemX[6] = domain.x(nd6i);
   elemX[7] = domain.x(nd7i);

   elemY[0] = domain.y(nd0i);
   elemY[1] = domain.y(nd1i);
   elemY[2] = domain.y(nd2i);
   elemY[3] = domain.y(nd3i);
   elemY[4] = domain.y(nd4i);
   elemY[5] = domain.y(nd5i);
   elemY[6] = domain.y(nd6i);
   elemY[7] = domain.y(nd7i);

   elemZ[0] = domain.z(nd0i);
   elemZ[1] = domain.z(nd1i);
   elemZ[2] = domain.z(nd2i);
   elemZ[3] = domain.z(nd3i);
   elemZ[4] = domain.z(nd4i);
   elemZ[5] = domain.z(nd5i);
   elemZ[6] = domain.z(nd6i);
   elemZ[7] = domain.z(nd7i);

}

/******************************************/

static inline
void InitStressTermsForElems(Domain &domain,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                             Index_t numElem)
{
   // pull in the stresses appropriate to the hydro integration
   // Récupérer les contraintes adaptées à l'intégration hydrodynamique
   // 提取适合当前流体力学积分步骤的应力值


   Kokkos::parallel_for(
      "InitStressTermsForElems",
      Kokkos::RangePolicy<>(0, numElem),
      KOKKOS_LAMBDA(const Index_t i) {
         sigxx[i] = sigyy[i] = sigzz[i] = - domain.p(i) - domain.q(i);
      });
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemShapeFunctionDerivatives( Real_t const x[],
                                              Real_t const y[],
                                              Real_t const z[],
                                              Real_t b[][8],
                                              Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors
     Calculer les cofacteurs
     计算体积雅可比矩阵的余子式（cofactor）
  */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
     Calculer les dérivées partielles : il suffit de le faire pour l = 0,1,2,3
     car, par symétrie, (6,7,4,5) = - (0,1,2,3).
     计算形函数导数：只需对结点 0,1,2,3 计算，因为由对称性可得
     结点 6,7,4,5 的导数是 0,1,2,3 的相反数。
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume)
     Calculer le déterminant du jacobien (volume)
     计算雅可比行列式，对应单元体积
  */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                              Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                              Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                              Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                              const Real_t x0, const Real_t y0, const Real_t z0,
                              const Real_t x1, const Real_t y1, const Real_t z1,
                              const Real_t x2, const Real_t y2, const Real_t z2,
                              const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemNodeNormals(Real_t pfx[8],
                                Real_t pfy[8],
                                Real_t pfz[8],
                                const Real_t x[8],
                                const Real_t y[8],
                                const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3
      Évaluer la première face : nœuds 0, 1, 2, 3
      计算第 1 个面：结点 0, 1, 2, 3
   */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1
      Évaluer la deuxième face : nœuds 0, 4, 5, 1
      计算第 2 个面：结点 0, 4, 5, 1
   */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2
      Évaluer la troisième face : nœuds 1, 5, 6, 2
      计算第 3 个面：结点 1, 5, 6, 2
   */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3
      Évaluer la quatrième face : nœuds 2, 6, 7, 3
      计算第 4 个面：结点 2, 6, 7, 3
   */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0
      Évaluer la cinquième face : nœuds 3, 7, 4, 0
      计算第 5 个面：结点 3, 7, 4, 0
   */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5
      Évaluer la sixième face : nœuds 4, 7, 6, 5
      计算第 6 个面：结点 4, 7, 6, 5
   */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void SumElemStressesToNodeForces( const Real_t B[][8],
                                         const Real_t stress_xx,
                                         const Real_t stress_yy,
                                         const Real_t stress_zz,
                                         Real_t fx[], Real_t fy[], Real_t fz[] )
{
   for(Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i] );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

static inline
void IntegrateStressForElems(const Domain &domain,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, Index_t numElem, Index_t /*tumNode*/)
{
   
   // loop over all elements (now in Kokkos)
   // Boucle sur tous les éléments (désormais avec Kokkos)
   // 对所有单元进行循环（使用 Kokkos 并行执行）

   Kokkos::parallel_for(
      "IntegrateStressForElems",
      Kokkos::RangePolicy<>(0, numElem),
      KOKKOS_LAMBDA(const Index_t k)
   {
      const Index_t* const elemToNode = domain.nodelist(k);
      Real_t B[3][8]; // shape function derivatives / dérivées des fonctions de forme / 形函数导数
      Real_t x_local[8];
      Real_t y_local[8];
      Real_t z_local[8];

      // get nodal coordinates from global arrays and copy into local arrays.
      // Récupérer les coordonnées nodales globales et les copier dans des tableaux locaux.
      // 从全局坐标数组中取出结点坐标，复制到局部数组中。
      CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

      // Volume calculation involves extra work for numerical consistency
      // Le calcul du volume demande un travail supplémentaire pour garantir la cohérence numérique.
      // 体积计算需要额外工作以保证数值一致性。
      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                       B, &determ[k]);

      // compute nodal normals based on shape derivatives
      // Calculer les normales nodales à partir des dérivées de forme.
      // 基于形函数导数计算结点法向量。
      CalcElemNodeNormals(B[0], B[1], B[2],
                          x_local, y_local, z_local);

      Real_t fx_local[8];
      Real_t fy_local[8];
      Real_t fz_local[8];

      // sum element stresses into nodal force contributions
      // Convertir les contraintes de l’élément en contributions de forces nodales.
      // 将单元应力转换为对各结点的力贡献。
      SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k],
                                  fx_local, fy_local, fz_local);

      // copy nodal force contributions to global force array with atomics
      // Copier les contributions nodales vers les forces globales en utilisant des opérations atomiques.
      // 使用原子加，将结点力贡献累加到全局力数组中。
      for (Index_t lnode = 0; lnode < 8; ++lnode) {
        Index_t gnode = elemToNode[lnode];
      	Kokkos::atomic_add(domain.fx_ptr() + gnode, fx_local[lnode]);
	Kokkos::atomic_add(domain.fy_ptr() + gnode, fy_local[lnode]);
	Kokkos::atomic_add(domain.fz_ptr() + gnode, fz_local[lnode]);

     }
   });
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
                    const Real_t x3, const Real_t x4, const Real_t x5,
                    const Real_t y0, const Real_t y1, const Real_t y2,
                    const Real_t y3, const Real_t y4, const Real_t y5,
                    const Real_t z0, const Real_t z1, const Real_t z2,
                    const Real_t z3, const Real_t z4, const Real_t z5,
                    Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemVolumeDerivative(Real_t dvdx[8],
                                     Real_t dvdy[8],
                                     Real_t dvdz[8],
                                     const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

/******************************************/
/* 
   Calcule la force anti-hourglass de Flanagan-Belytschko pour un élément.
   计算 Flanagan–Belytschko 反沙漏力（Hourglass Force）
*/
KOKKOS_INLINE_FUNCTION
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                              Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz)
{
   Real_t hxx[4];

   /* Calcul de la composante x des modes hourglass
      计算 hourglass 模式的 x 分量 */
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }

   /* Contribution à la force en x
      累积 x 方向力 */
   for(Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient * (hourgam[i][0] * hxx[0] +
                               hourgam[i][1] * hxx[1] +
                               hourgam[i][2] * hxx[2] +
                               hourgam[i][3] * hxx[3]);
   }

   /* Mode y
      计算 y 模式 */
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient * (hourgam[i][0] * hxx[0] +
                               hourgam[i][1] * hxx[1] +
                               hourgam[i][2] * hxx[2] +
                               hourgam[i][3] * hxx[3]);
   }

   /* Mode z
      计算 z 模式 */
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }

   for(Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient * (hourgam[i][0] * hxx[0] +
                               hourgam[i][1] * hxx[1] +
                               hourgam[i][2] * hxx[2] +
                               hourgam[i][3] * hxx[3]);
   }
}

/* 
   Calcule la force anti-hourglass (Flanagan–Belytschko)
   计算 Flanagan–Belytschko 反沙漏力（Hourglass Force）
*/
void CalcFBHourglassForceForElems(Domain& domain,
                                  Real_t* determ,
                                  Real_t* x8n, Real_t* y8n, Real_t* z8n,
                                  Real_t* dvdx, Real_t* dvdy, Real_t* dvdz,
                                  Real_t hourg,
                                  Index_t numElem,
                                  Index_t numNode)
{
   /* 
      Tableaux gamma : modes hourglass de référence.
      gamma 数组：标准 hourglass 模式
   */
   Real_t gamma[4][8] = {
      { 1.,  1., -1., -1., -1., -1.,  1.,  1.},
      { 1., -1., -1.,  1., -1.,  1.,  1., -1.},
      { 1., -1.,  1., -1.,  1., -1.,  1., -1.},
      {-1.,  1., -1.,  1.,  1., -1.,  1., -1.}
   };

   /* 
      Boucle parallèle sur les éléments
      对所有单元执行并行计算
   */
   Kokkos::parallel_for(
      "CalcFBHourglassForceForElems",
      Kokkos::RangePolicy<>(0, numElem),
      KOKKOS_LAMBDA(const Index_t i2)
   {
      Real_t hourgam[8][4];
      Real_t xd1[8], yd1[8], zd1[8];
      Real_t hgfx[8], hgfy[8], hgfz[8];

      const Index_t* elemToNode = domain.nodelist(i2);
      const Index_t base = i2 * 8;

      Real_t volinv = Real_t(1.0) / determ[i2];

      /* 
         Calcule les modes hourglass modifiés.
         计算修正后的 hourglass 模式 hourgam
      */
      for (Index_t m = 0; m < 4; ++m) {

         Real_t hourmodx =
            x8n[base+0] * gamma[m][0] + x8n[base+1] * gamma[m][1] +
            x8n[base+2] * gamma[m][2] + x8n[base+3] * gamma[m][3] +
            x8n[base+4] * gamma[m][4] + x8n[base+5] * gamma[m][5] +
            x8n[base+6] * gamma[m][6] + x8n[base+7] * gamma[m][7];

         Real_t hourmody =
            y8n[base+0] * gamma[m][0] + y8n[base+1] * gamma[m][1] +
            y8n[base+2] * gamma[m][2] + y8n[base+3] * gamma[m][3] +
            y8n[base+4] * gamma[m][4] + y8n[base+5] * gamma[m][5] +
            y8n[base+6] * gamma[m][6] + y8n[base+7] * gamma[m][7];

         Real_t hourmodz =
            z8n[base+0] * gamma[m][0] + z8n[base+1] * gamma[m][1] +
            z8n[base+2] * gamma[m][2] + z8n[base+3] * gamma[m][3] +
            z8n[base+4] * gamma[m][4] + z8n[base+5] * gamma[m][5] +
            z8n[base+6] * gamma[m][6] + z8n[base+7] * gamma[m][7];

         for (Index_t n = 0; n < 8; ++n) {
            hourgam[n][m] =
                 gamma[m][n]
               - volinv * ( dvdx[base+n] * hourmodx +
                            dvdy[base+n] * hourmody +
                            dvdz[base+n] * hourmodz );
         }
      }

      /* 
         Charge les vitesses nodales
         加载节点速度 xd/yd/zd
      */
      for(Index_t n = 0; n < 8; ++n) {
         Index_t g = elemToNode[n];
         xd1[n] = domain.xd(g);
         yd1[n] = domain.yd(g);
         zd1[n] = domain.zd(g);
      }

      /* 
         coefficient = - hourg * 0.01 * ss * mass / volume^(1/3)
         计算力系数
      */
      Real_t ss1     = domain.ss(i2);
      Real_t mass1   = domain.elemMass(i2);
      Real_t volume13 = CBRT(determ[i2]);
      Real_t coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      /* 
         Calcule les forces hourglass sur les 8 nœuds
         计算 8 个节点的沙漏力
      */
      CalcElemFBHourglassForce(xd1, yd1, zd1,
                               hourgam,
                               coefficient,
                               hgfx, hgfy, hgfz);

      /* 
         Ajoute les forces à chaque nœud
         累积节点的力（atomic 以避免并发写）
      */
      for(Index_t n = 0; n < 8; ++n) {
         Index_t g = elemToNode[n];
        Kokkos::atomic_add(domain.fx_ptr() + g, hgfx[n]);
	Kokkos::atomic_add(domain.fy_ptr() + g, hgfy[n]);
	Kokkos::atomic_add(domain.fz_ptr() + g, hgfz[n]);

      }
   });
}

/* 
   Calcule les dérivées de volume et prépare le contrôle hourglass.
   计算体积导数并为 hourglass 控制做准备
*/
void CalcHourglassControlForElems(Domain& domain,
                                  Real_t determ[],
                                  Real_t hgcoef)
{
   Index_t numElem  = domain.numElem();
   Index_t numElem8 = numElem * 8;

   /* 
      Allocation des tableaux temporaires (host ou device selon implémentation)
      分配临时数组（遵从你的 Allocate/Release 接口）
   */
   Real_t* dvdx = Allocate<Real_t>(numElem8);
   Real_t* dvdy = Allocate<Real_t>(numElem8);
   Real_t* dvdz = Allocate<Real_t>(numElem8);

   Real_t* x8n  = Allocate<Real_t>(numElem8);
   Real_t* y8n  = Allocate<Real_t>(numElem8);
   Real_t* z8n  = Allocate<Real_t>(numElem8);

   /* 
      Boucle parallèle : calcule pfx/pfy/pfz et les coordonnées locales x8n/y8n/z8n
      并行循环：计算体积导数并填充 x8n/y8n/z8n
   */
   Kokkos::parallel_for(
      "CalcHourglassControlForElems",
      Kokkos::RangePolicy<>(0, numElem),
      KOKKOS_LAMBDA(const Index_t i)
   {
      Real_t x1[8], y1[8], z1[8];
      Real_t pfx[8], pfy[8], pfz[8];

      const Index_t* elemToNode = domain.nodelist(i);

      /* 
         Charge les coordonnées des 8 nœuds dans des tableaux locaux.
         加载 8 节点坐标到局部数组
      */
      CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

      /* 
         Calcule les dérivées du volume pour les 8 nœuds
         计算 8 个节点的体积导数
      */
      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* 
         Stockage dans les tableaux temporaires dvdx, dvdy, dvdz, x8n, y8n, z8n
         将结果写入临时数组
      */
      for (Index_t n = 0; n < 8; ++n) {
         Index_t idx = 8 * i + n;
         dvdx[idx] = pfx[n];
         dvdy[idx] = pfy[n];
         dvdz[idx] = pfz[n];

         x8n[idx]  = x1[n];
         y8n[idx]  = y1[n];
         z8n[idx]  = z1[n];
      }

      /* 
         determ[i] = vol0 * v
         储存初始体积（用于 hourglass 计算）
      */
      determ[i] = domain.volo(i) * domain.v(i);

      /* Vérification des volumes négatifs
         检查负体积 */
      if (domain.v(i) <= Real_t(0.0)) {
         exit(VolumeError);
      }
   });

   /* 
      Si le coefficient hourglass > 0, appliquer le calcul FB hourglass
      若 hgcoef > 0，则调用 FB Hourglass 力方法
   */
   if (hgcoef > Real_t(0.0)) {
      CalcFBHourglassForceForElems(
         domain,
         determ,
         x8n, y8n, z8n,
         dvdx, dvdy, dvdz,
         hgcoef,
         numElem,
         domain.numNode()
      );
   }

   /* 
      Libération de la mémoire temporaire
      释放临时内存
   */
   Release(&z8n);
   Release(&y8n);
   Release(&x8n);
   Release(&dvdz);
   Release(&dvdy);
   Release(&dvdx);
}
/* 
   Calcule les forces de volume pour tous les éléments.
   计算所有单元的体积力（包括应力部分 + 沙漏部分）
*/
void CalcVolumeForceForElems(Domain& domain)
{
   Index_t numElem = domain.numElem();

   if (numElem == 0) {
      return;
   }

   Real_t hgcoef = domain.hgcoef();

   /* 
      Allocation des tableaux de contraintes et volumes
      分配应力与体积数组
   */
   Real_t* sigxx  = Allocate<Real_t>(numElem);
   Real_t* sigyy  = Allocate<Real_t>(numElem);
   Real_t* sigzz  = Allocate<Real_t>(numElem);
   Real_t* determ = Allocate<Real_t>(numElem);

   /* 
      Initialise les termes de contraintes sigxx/sigyy/sigzz
      初始化应力项（sigxx/sigyy/sigzz）
   */
   InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

   /* 
      Intégration des contraintes dans les forces nodales
      将应力积分转化为节点力 (Volume + deviatoric)
   */
   IntegrateStressForElems(
         domain,
         sigxx, sigyy, sigzz,
         determ,
         numElem,
         domain.numNode()
      );

   /* 
      Vérification du volume négatif
      检查是否出现负体积
   */
   Kokkos::parallel_for(
      "CheckElemVolume",
      Kokkos::RangePolicy<>(0, numElem),
      KOKKOS_LAMBDA(const Index_t k)
   {
      if (determ[k] <= Real_t(0.0)) {
         exit(VolumeError);
      }
   });

   /* 
      Contrôle hourglass si nécessaire
      如需要则进行沙漏控制
   */
   CalcHourglassControlForElems(domain, determ, hgcoef);

   /* 
      Libération mémoire
      释放内存
   */
   Release(&determ);
   Release(&sigzz);
   Release(&sigyy);
   Release(&sigxx);
}

/* 
   Calcule les forces totales pour tous les nœuds.
   计算所有节点的总力（包括单元体积力 + 沙漏力）
*/
void CalcForceForNodes(Domain& domain)
{
   Index_t numNode = domain.numNode();

#if USE_MPI
   /* 
      Réception des données fantômes (ghost nodes) pour les forces.
      接收 ghost 节点力信息（MPI，接口保留）
   */
   CommRecv(domain, MSG_COMM_SBN, 3,
            domain.sizeX() + 1,
            domain.sizeY() + 1,
            domain.sizeZ() + 1,
            true, false);
#endif

   /* 
      Mise à zéro des forces nodales fx, fy, fz
      将所有节点力 fx/fy/fz 清零
   */
   auto fx = domain.fx_view();
   auto fy = domain.fy_view();
   auto fz = domain.fz_view();
   Kokkos::parallel_for(
      "ZeroNodeForces",
      Kokkos::RangePolicy<>(0, numNode),
      KOKKOS_LAMBDA(const Index_t i)
   {
      fx(i) = Real_t(0.0);
      fy(i) = Real_t(0.0);
      fz(i) = Real_t(0.0);
   });

   /* 
      Calcul des forces de volume (stress + hourglass)
      调用体积力（应力 + 沙漏控制）的计算
   */
   CalcVolumeForceForElems(domain);

#if USE_MPI
   /*
      Préparation des champs à envoyer：fx, fy, fz
      准备要发送的力场数据：fx/fy/fz
   */
   Domain_member fieldData[3];
   fieldData[0] = &Domain::fx;
   fieldData[1] = &Domain::fy;
   fieldData[2] = &Domain::fz;

   /* 
      Envoi des données vers les voisins MPI.
      向 MPI 邻居发送节点力（接口保留）
   */
   CommSend(domain, MSG_COMM_SBN, 3, fieldData,
            domain.sizeX() + 1,
            domain.sizeY() + 1,
            domain.sizeZ() + 1,
            true, false);

   /* 
      Fusion des forces fantômes dans les points frontières.
      合并 ghost 节点的力
   */
   CommSBN(domain, 3, fieldData);
#endif
}

/*
   Calcule l’accélération des nœuds à partir des forces nodales.
   根据节点力计算节点加速度（xdd, ydd, zdd）
*/

void CalcAccelerationForNodes(Domain &domain, Index_t numNode)
{
   // ① 先把 Domain 里的 View 拿出来（普通 C++ 语句）
   auto xdd = domain.xdd_view();
   auto ydd = domain.ydd_view();
   auto zdd = domain.zdd_view();

   auto fx  = domain.fx_view();
   auto fy  = domain.fy_view();
   auto fz  = domain.fz_view();

   auto nodalMass = domain.nodalMass_view();

   // ② 再调用 Kokkos::parallel_for
   Kokkos::parallel_for(
      "CalcAccelerationForNodes",
      Kokkos::RangePolicy<>(0, numNode),
      KOKKOS_LAMBDA(const Index_t i)
      {
         xdd(i) = fx(i) / nodalMass(i);
         ydd(i) = fy(i) / nodalMass(i);
         zdd(i) = fz(i) / nodalMass(i);
      }
   );
}


/*
   Applique les conditions limites d’accélération sur les nœuds
   对节点应用加速度边界条件（固定方向加速度 = 0）
*/
void ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
{
   auto xdd = domain.xdd_view();
   auto ydd = domain.ydd_view();
   auto zdd = domain.zdd_view();

   auto symmX = domain.symmX_view();
   auto symmY = domain.symmY_view();
   auto symmZ = domain.symmZ_view();

   Index_t size = domain.sizeX();
   Index_t numNodeBC = (size + 1) * (size + 1);

   /* 
      BC en X : si la liste n’est pas vide, mettre xdd = 0
      X方向边界条件：如果边界列表非空，则将所有这些节点的 xdd 设为 0
   */
   if (!domain.symmXempty()) {
      Kokkos::parallel_for(
         "BC_X",
         Kokkos::RangePolicy<>(0, numNodeBC),
         KOKKOS_LAMBDA(const Index_t i)
      {
         xdd(domain.symmX(i)) = Real_t(0.0);
      });
   }

   /*
      BC en Y : mettre ydd = 0 sur la frontière Y
      Y方向边界条件：将 Y 边界节点的 ydd 设为 0
   */
   if (!domain.symmYempty()) {
      Kokkos::parallel_for(
         "BC_Y",
         Kokkos::RangePolicy<>(0, numNodeBC),
         KOKKOS_LAMBDA(const Index_t i)
      {
         ydd(domain.symmY(i)) = Real_t(0.0);
      });
   }

   /*
      BC en Z : mettre zdd = 0 sur la frontière Z
      Z方向边界条件：将 Z 边界节点的 zdd 设为 0
   */
   if (!domain.symmZempty()) {
      Kokkos::parallel_for(
         "BC_Z",
         Kokkos::RangePolicy<>(0, numNodeBC),
         KOKKOS_LAMBDA(const Index_t i)
      {
         zdd(domain.symmZ(i)) = Real_t(0.0);
      });
   }
}

/*
   Met à jour la vitesse des nœuds en utilisant l’accélération.
   根据加速度更新节点速度（xd, yd, zd）
   Applique également un seuil u_cut：若速度幅度太小则归零。
*/
void CalcVelocityForNodes(Domain &domain,
                          const Real_t dt,
                          const Real_t u_cut,
                          Index_t numNode)
{
// --- 提前取出 Views（这是关键）
    auto xd  = domain.xd_view();
    auto yd  = domain.yd_view();
    auto zd  = domain.zd_view();

    auto xdd = domain.xdd_view();
    auto ydd = domain.ydd_view();
    auto zdd = domain.zdd_view();

    Kokkos::parallel_for(
      "CalcVelocityForNodes",
      Kokkos::RangePolicy<>(0, numNode),
      KOKKOS_LAMBDA(const Index_t i)
   {
      /* mise à jour de la vitesse x : vx_new = vx_old + ax * dt
         更新 x 方向速度：xd = xd + xdd * dt
      */
      Real_t xdtmp = xd(i) + xdd(i) * dt;

      // seuil u_cut : si trop petit, mettre à zéro
      // 速度绝对值若小于 u_cut，则认为为 0（避免震荡）
      if (FABS(xdtmp) < u_cut) {
         xdtmp = Real_t(0.0);
      }
      xd(i) = xdtmp;

      /* idem pour la vitesse y */
      Real_t ydtmp = yd(i) + ydd(i) * dt;
      if (FABS(ydtmp) < u_cut) {
         ydtmp = Real_t(0.0);
      }
      yd(i) = ydtmp;

      /* idem pour la vitesse z */
      Real_t zdtmp = zd(i) + zdd(i) * dt;
      if (FABS(zdtmp) < u_cut) {
         zdtmp = Real_t(0.0);
      }
      zd(i) = zdtmp;
   });
}

/*
   Met à jour la position des nœuds en utilisant la vitesse.
   根据节点速度更新节点位置（x, y, z）
*/
void CalcPositionForNodes(Domain &domain,
                          const Real_t dt,
                          Index_t numNode)
{
// --- 提前取出 Views
   auto x  = domain.x_view();
   auto y  = domain.y_view();
   auto z  = domain.z_view();

   auto xd = domain.xd_view();
   auto yd = domain.yd_view();
   auto zd = domain.zd_view();

   Kokkos::parallel_for(
      "CalcPositionForNodes",
      Kokkos::RangePolicy<>(0, numNode),
      KOKKOS_LAMBDA(const Index_t i)
   {
      /* x_new = x_old + vx * dt
         根据速度更新 x 坐标
      */
      x(i) += xd(i) * dt;

      /* y_new = y_old + vy * dt
         更新 y 坐标
      */
      y(i) += yd(i) * dt;

      /* z_new = z_old + vz * dt
         更新 z 坐标
      */
      z(i) += zd(i) * dt;
   });
}

/*
   Effectue l’étape nodale lagrangienne : force → accélération → vitesse → position.
   执行拉格朗日节点更新步骤：力 → 加速度 → 速度 → 位置
*/
KOKKOS_INLINE_FUNCTION
void LagrangeNodal(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   Domain_member fieldData[6] ;
#endif

   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

   /* 
      Les conditions limites sont évaluées au début : d'abord force puis accélération.
      边界条件在时间步开始时应用：先计算力，再计算加速度
   */
   CalcForceForNodes(domain);

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain.sizeX()+1, domain.sizeY()+1, domain.sizeZ()+1,
            false, false);
#endif
#endif

   /* Calcul de l’accélération nodale
      计算节点加速度
   */
   CalcAccelerationForNodes(domain, domain.numNode());

   /* Application des conditions limites d’accélération
      应用加速度边界条件
   */
   ApplyAccelerationBoundaryConditionsForNodes(domain);

   /* Mise à jour de la vitesse nodale
      更新节点速度
   */
   CalcVelocityForNodes(domain, delt, u_cut, domain.numNode());

   /* Mise à jour de la position nodale
      更新节点位置
   */
   CalcPositionForNodes(domain, delt, domain.numNode());

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain.sizeX()+1, domain.sizeY()+1, domain.sizeZ()+1,
            false, false);
   CommSyncPosVel(domain);
#endif
#endif
}

/*
   Calcule le volume d’un élément hexaédrique à 8 nœuds.
   计算 8 节点六面体单元的体积
*/
KOKKOS_INLINE_FUNCTION
Real_t CalcElemVolume(
    const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
    const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
    const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
    const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
    const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
    const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7)
{
    /* coefficient 1/12
       1/12 系数（六面体体积计算公式中的常数）
    */
    const Real_t twelveth = Real_t(1.0) / Real_t(12.0);

    /* 
       Differences entre différents couples de nœuds
       计算节点差向量
    */
    Real_t dx61 = x6 - x1;
    Real_t dy61 = y6 - y1;
    Real_t dz61 = z6 - z1;

    Real_t dx70 = x7 - x0;
    Real_t dy70 = y7 - y0;
    Real_t dz70 = z7 - z0;

    Real_t dx63 = x6 - x3;
    Real_t dy63 = y6 - y3;
    Real_t dz63 = z6 - z3;

    Real_t dx20 = x2 - x0;
    Real_t dy20 = y2 - y0;
    Real_t dz20 = z2 - z0;

    Real_t dx50 = x5 - x0;
    Real_t dy50 = y5 - y0;
    Real_t dz50 = z5 - z0;

    Real_t dx64 = x6 - x4;
    Real_t dy64 = y6 - y4;
    Real_t dz64 = z6 - z4;

    Real_t dx31 = x3 - x1;
    Real_t dy31 = y3 - y1;
    Real_t dz31 = z3 - z1;

    Real_t dx72 = x7 - x2;
    Real_t dy72 = y7 - y2;
    Real_t dz72 = z7 - z2;

    Real_t dx43 = x4 - x3;
    Real_t dy43 = y4 - y3;
    Real_t dz43 = z4 - z3;

    Real_t dx57 = x5 - x7;
    Real_t dy57 = y5 - y7;
    Real_t dz57 = z5 - z7;

    Real_t dx14 = x1 - x4;
    Real_t dy14 = y1 - y4;
    Real_t dz14 = z1 - z4;

    Real_t dx25 = x2 - x5;
    Real_t dy25 = y2 - y5;
    Real_t dz25 = z2 - z5;

    /*
       Produit triple : détermine 6 fois le volume du tétraèdre.
       三重积：计算三个向量的体积贡献
    */
    auto TRIPLE = [&](Real_t x1, Real_t y1, Real_t z1,
                      Real_t x2, Real_t y2, Real_t z2,
                      Real_t x3, Real_t y3, Real_t z3) -> Real_t
    {
        return x1 * (y2 * z3 - z2 * y3)
             + x2 * (z1 * y3 - y1 * z3)
             + x3 * (y1 * z2 - z1 * y2);
    };

    /*
       Volume total = somme de trois produits triples
       总体积 = 三个三重积之和
    */
    Real_t volume =
        TRIPLE(dx31 + dx72, dy31 + dy72, dz31 + dz72,
               dx63, dy63, dz63,
               dx20, dy20, dz20)

      + TRIPLE(dx43 + dx57, dy43 + dy57, dz43 + dz57,
               dx64, dy64, dz64,
               dx70, dy70, dz70)

      + TRIPLE(dx14 + dx25, dy14 + dy25, dz14 + dz25,
               dx61, dy61, dz61,
               dx50, dy50, dz50);

    /* normalisation par 1/12
       最终体积须乘以 1/12
    */
    return volume * twelveth;
}

/*
   Calcule une mesure géométrique associée à une face quadrilatérale.
   计算四边形单元面的一种几何度量（用于特征长度计算）
*/
KOKKOS_INLINE_FUNCTION
Real_t AreaFace(
    const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
    const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
    const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3)
{
    /*
       fx, fy, fz = (x2 - x0) − (x3 - x1)
       第一个向量差：表示四边形的一条对角线方向差值
    */
    Real_t fx = (x2 - x0) - (x3 - x1);
    Real_t fy = (y2 - y0) - (y3 - y1);
    Real_t fz = (z2 - z0) - (z3 - z1);

    /*
       gx, gy, gz = (x2 - x0) + (x3 - x1)
       第二个向量和：另一条方向的组合
    */
    Real_t gx = (x2 - x0) + (x3 - x1);
    Real_t gy = (y2 - y0) + (y3 - y1);
    Real_t gz = (z2 - z0) + (z3 - z1);

    /*
       L’expression combine deux termes : ||f||^2 * ||g||^2 − (f·g)^2
       该表达式计算 f 与 g 的相关几何量：||f||^2 * ||g||^2 − (f·g)^2
       这是平行四边形面积平方的等价表示
    */
    Real_t term1 = (fx*fx + fy*fy + fz*fz) * (gx*gx + gy*gy + gz*gz);
    Real_t term2 = (fx*gx + fy*gy + fz*gz);

    return term1 - term2 * term2;
}

/*
   Calcule la longueur caractéristique d’un élément hexaédrique.
   计算六面体单元的特征长度（用于时间步长、人工粘性等）
*/
KOKKOS_INLINE_FUNCTION
Real_t CalcElemCharacteristicLength(
    const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8],
    const Real_t volume)
{
    /* 
       charLength sera le plus grand area_face parmi les 6 faces.
       charLength 将存储六个面的最大 areaFace 值（面积相关量）
    */
    Real_t charLength = Real_t(0.0);

    /* face avant (0,1,2,3)
       面 1：节点 0,1,2,3
    */
    {
        Real_t a = AreaFace(
            x[0], x[1], x[2], x[3],
            y[0], y[1], y[2], y[3],
            z[0], z[1], z[2], z[3]);
        charLength = (a > charLength ? a : charLength);
    }

    /* face arrière (4,5,6,7)
       面 2：节点 4,5,6,7
    */
    {
        Real_t a = AreaFace(
            x[4], x[5], x[6], x[7],
            y[4], y[5], y[6], y[7],
            z[4], z[5], z[6], z[7]);
        charLength = (a > charLength ? a : charLength);
    }

    /* face latérale (0,1,5,4)
       面 3：节点 0,1,5,4
    */
    {
        Real_t a = AreaFace(
            x[0], x[1], x[5], x[4],
            y[0], y[1], y[5], y[4],
            z[0], z[1], z[5], z[4]);
        charLength = (a > charLength ? a : charLength);
    }

    /* face latérale (1,2,6,5)
       面 4：节点 1,2,6,5
    */
    {
        Real_t a = AreaFace(
            x[1], x[2], x[6], x[5],
            y[1], y[2], y[6], y[5],
            z[1], z[2], z[6], z[5]);
        charLength = (a > charLength ? a : charLength);
    }

    /* face latérale (2,3,7,6)
       面 5：节点 2,3,7,6
    */
    {
        Real_t a = AreaFace(
            x[2], x[3], x[7], x[6],
            y[2], y[3], y[7], y[6],
            z[2], z[3], z[7], z[6]);
        charLength = (a > charLength ? a : charLength);
    }

    /* face latérale (3,0,4,7)
       面 6：节点 3,0,4,7
    */
    {
        Real_t a = AreaFace(
            x[3], x[0], x[4], x[7],
            y[3], y[0], y[4], y[7],
            z[3], z[0], z[4], z[7]);
        charLength = (a > charLength ? a : charLength);
    }

    /*
       charLength = 4 * volume / sqrt(max_area)
       公式：特征长度 = 4 * 体积 / 面积平方根
       （来源于 LULESH 论文，用于 CFL 时间步限制）
    */
    charLength = Real_t(4.0) * volume / sqrt(charLength);

    return charLength;
}

/*
   Calcule le gradient de vitesse (D) à l’intérieur d’un élément.
   计算单元内部的速度梯度张量（应变速率张量 D）
   Entrées:
     xvel[8], yvel[8], zvel[8]  → vitesses nodales / 节点速度
     b[3][8]                    → dérivées des fonctions de forme / 形函数导数
     detJ                       → déterminant jacobien / 雅可比行列式
   Sortie:
     d[6] → {dxx, dyy, dzz, dyz, dxz, dxy}
*/
KOKKOS_INLINE_FUNCTION
void CalcElemVelocityGradient(
    const Real_t* const xvel,
    const Real_t* const yvel,
    const Real_t* const zvel,
    const Real_t b[][8],
    const Real_t detJ,
    Real_t* const d )
{
    const Real_t inv_detJ = Real_t(1.0) / detJ;

    Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;

    const Real_t* const pfx = b[0];  /* dérivées ∂N/∂x  形函数 x 方向导数 */
    const Real_t* const pfy = b[1];  /* dérivées ∂N/∂y */
    const Real_t* const pfz = b[2];  /* dérivées ∂N/∂z */

    /* 
       dxx, dyy, dzz = gradient diagonal
       计算速度梯度张量的对角项
    */
    d[0] = inv_detJ * ( pfx[0]*(xvel[0]-xvel[6])
                      + pfx[1]*(xvel[1]-xvel[7])
                      + pfx[2]*(xvel[2]-xvel[4])
                      + pfx[3]*(xvel[3]-xvel[5]) );

    d[1] = inv_detJ * ( pfy[0]*(yvel[0]-yvel[6])
                      + pfy[1]*(yvel[1]-yvel[7])
                      + pfy[2]*(yvel[2]-yvel[4])
                      + pfy[3]*(yvel[3]-yvel[5]) );

    d[2] = inv_detJ * ( pfz[0]*(zvel[0]-zvel[6])
                      + pfz[1]*(zvel[1]-zvel[7])
                      + pfz[2]*(zvel[2]-zvel[4])
                      + pfz[3]*(zvel[3]-zvel[5]) );

    /* Off-diagonal terms
       非对角项的组合运算
    */
    dyddx = inv_detJ * ( pfx[0]*(yvel[0]-yvel[6])
                       + pfx[1]*(yvel[1]-yvel[7])
                       + pfx[2]*(yvel[2]-yvel[4])
                       + pfx[3]*(yvel[3]-yvel[5]) );

    dxddy = inv_detJ * ( pfy[0]*(xvel[0]-xvel[6])
                       + pfy[1]*(xvel[1]-xvel[7])
                       + pfy[2]*(xvel[2]-xvel[4])
                       + pfy[3]*(xvel[3]-xvel[5]) );

    dzddx = inv_detJ * ( pfx[0]*(zvel[0]-zvel[6])
                       + pfx[1]*(zvel[1]-zvel[7])
                       + pfx[2]*(zvel[2]-zvel[4])
                       + pfx[3]*(zvel[3]-zvel[5]) );

    dxddz = inv_detJ * ( pfz[0]*(xvel[0]-xvel[6])
                       + pfz[1]*(xvel[1]-xvel[7])
                       + pfz[2]*(xvel[2]-xvel[4])
                       + pfz[3]*(xvel[3]-xvel[5]) );

    dzddy = inv_detJ * ( pfy[0]*(zvel[0]-zvel[6])
                       + pfy[1]*(zvel[1]-zvel[7])
                       + pfy[2]*(zvel[2]-zvel[4])
                       + pfy[3]*(zvel[3]-zvel[5]) );

    dyddz = inv_detJ * ( pfz[0]*(yvel[0]-yvel[6])
                       + pfz[1]*(yvel[1]-yvel[7])
                       + pfz[2]*(yvel[2]-yvel[4])
                       + pfz[3]*(yvel[3]-yvel[5]) );

    /*
       dxy, dxz, dyz = combinaisons symétriques
       对称处理（应变速率张量）
    */
    d[5] = Real_t(0.5) * (dxddy + dyddx); /* dxy */
    d[4] = Real_t(0.5) * (dxddz + dzddx); /* dxz */
    d[3] = Real_t(0.5) * (dzddy + dyddz); /* dyz */
}

/* 
   Calcule la cinématique des éléments : volume, volume relatif, 
   vitesse gradient, longueur caractéristique.
   计算单元的运动学量：体积、相对体积、速度梯度、特征长度。
*/
KOKKOS_INLINE_FUNCTION
void CalcKinematicsForElems(
    Domain &domain,
    const Real_t deltaTime,
    const Index_t numElem)
{
    /* 
       Boucle sur tous les éléments.
       对所有单元执行循环。
    */
    // --- 取出需要写的 Views（关键）
auto vnew   = domain.vnew_view();
auto delv   = domain.delv_view();
auto arealg = domain.arealg_view();

auto dxx    = domain.dxx_view();
auto dyy    = domain.dyy_view();
auto dzz    = domain.dzz_view();

    Kokkos::parallel_for(
        "CalcKinematicsForElems",
        Kokkos::RangePolicy<>(0, numElem),
        KOKKOS_LAMBDA(const Index_t k)
    {
        Real_t B[3][8];   /* dérivées des fonctions de forme / 形函数导数 */
        Real_t D[6];      /* gradient de vitesse / 速度梯度张量 */

        Real_t x_local[8];
        Real_t y_local[8];
        Real_t z_local[8];

        Real_t xd_local[8];
        Real_t yd_local[8];
        Real_t zd_local[8];

        Real_t detJ = Real_t(0.0);

        const Index_t* const elemToNode = domain.nodelist(k);

        /* ---------------------------------------------------------
           1. Charger les coordonnées nodales
           1. 收集单元节点坐标
        --------------------------------------------------------- */
        CollectDomainNodesToElemNodes(domain, elemToNode,
                                      x_local, y_local, z_local);

        /* ---------------------------------------------------------
           2. Calcul du volume de l’élément
           2. 计算单元体积
        --------------------------------------------------------- */
        Real_t volume = CalcElemVolume(x_local, y_local, z_local);
        Real_t relativeVolume = volume / domain.volo(k);

        vnew(k) = relativeVolume;           /* nouveau volume 相对体积 */
        delv(k) = relativeVolume - domain.v(k); /* 体积变化 */

        /* ---------------------------------------------------------
           3. Longueur caractéristique (géométrie)
           3. 单元特征长度
        --------------------------------------------------------- */
        arealg(k) =
            CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

        /* ---------------------------------------------------------
           4. Charger vitesses nodales
           4. 加载节点速度
        --------------------------------------------------------- */
        for (Index_t ln = 0; ln < 8; ++ln) {
            Index_t g = elemToNode[ln];
            xd_local[ln] = domain.xd(g);
            yd_local[ln] = domain.yd(g);
            zd_local[ln] = domain.zd(g);
        }

        /* ---------------------------------------------------------
           5. Appliquer un décalage demi-pas (Lagrange)
              x -= (Δt/2) * v
           5. Lagrange 半步推进
        --------------------------------------------------------- */
        Real_t dt2 = Real_t(0.5) * deltaTime;
        for (Index_t j = 0; j < 8; ++j) {
            x_local[j] -= dt2 * xd_local[j];
            y_local[j] -= dt2 * yd_local[j];
            z_local[j] -= dt2 * zd_local[j];
        }

        /* ---------------------------------------------------------
           6. Calculer B 矩阵 + detJ（雅可比行列式）
        --------------------------------------------------------- */
        CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &detJ);

        /* ---------------------------------------------------------
           7. Calculer le gradient de vitesse D
           7. 计算速度梯度张量 D
        --------------------------------------------------------- */
        CalcElemVelocityGradient(xd_local, yd_local, zd_local,
                                 B, detJ, D);

        /* ---------------------------------------------------------
           8. Stocker les quantités dans le domain
           8. 写回域数据
        --------------------------------------------------------- */
        dxx(k) = D[0];
        dyy(k) = D[1];
        dzz(k) = D[2];
    });
}

/*
   Met à jour les grandeurs lagrangiennes des éléments :
   calcul du taux de déformation, contrainte de déviateur, 
   et vérification du volume.
   更新单元的拉格朗日物理量：应变速率、偏应变，以及体积检查。
*/
KOKKOS_INLINE_FUNCTION
void CalcLagrangeElements(Domain& domain)
{
    Index_t numElem = domain.numElem();

    if (numElem > 0) {

        const Real_t deltatime = domain.deltatime();

        /* 
           Allouer les tableaux de déformation (si nécessaire)
           分配应变相关数组（如需要）
        */
        domain.AllocateStrains(numElem);

        /* 
           Calcul des grandeurs cinématiques par élément
           计算每个单元的运动学量
        */
        CalcKinematicsForElems(domain, deltatime, numElem);

        /* 
           Deuxième boucle : imposer la contrainte déviatorique,
           calculer vdov, et détecter les volumes négatifs.
           第二个循环：施加偏应变约束、计算vdov、检测体积负值。
        */

auto dxx  = domain.dxx_view();
auto dyy  = domain.dyy_view();
auto dzz  = domain.dzz_view();
auto vdov_v = domain.vdov_view();
auto vnew = domain.vnew_view();


        Kokkos::parallel_for(
            "CalcLagrangeElements",
            Kokkos::RangePolicy<>(0, numElem),
            KOKKOS_LAMBDA(const Index_t k)
        {
            /* 
               Taux de déformation volumique vdov = tr(D)
               体积应变速率 vdov = 速度梯度迹（Dxx+Dyy+Dzz）
            */
            Real_t vdov = domain.dxx(k)
                        + domain.dyy(k)
                        + domain.dzz(k);

            Real_t vdovthird = vdov / Real_t(3.0);

            /* 
               Rendre le tenseur de déformation déviatorique
               使变形张量成为偏张量（去除体积部分）
            */
            vdov_v(k) = vdov;

            dxx(k) -= vdovthird;
            dyy(k) -= vdovthird;
            dzz(k) -= vdovthird;

            /* 
               Vérification du volume : si vnew ≤ 0 → erreur fatale
               检查体积：若 vnew ≤ 0 → 致命错误
            */
            if (vnew(k) <= Real_t(0.0)) {
                /* 
                   Version MPI retirée (USE_MPI=0).
                   删除 MPI 版本（USE_MPI=0）
                */
                Kokkos::abort("VolumeError: negative element volume detected.");
            }
        });

        /* 
           Libérer les tableaux temporaires de déformation
           释放应变的临时数组
        */
        domain.DeallocateStrains();
    }
}

/*
   Calcule les gradients nécessaires pour le Q monotone (delx, delv)
   计算单调人工粘性（Q）的梯度信息：delx_* 与 delv_*
*/
KOKKOS_INLINE_FUNCTION
void CalcMonotonicQGradientsForElems(Domain& domain)
{
    const Index_t numElem = domain.numElem();

// --- node-based views
auto x  = domain.x_view();
auto y  = domain.y_view();
auto z  = domain.z_view();

auto xd = domain.xd_view();
auto yd = domain.yd_view();
auto zd = domain.zd_view();

// --- elem-based views
auto volo = domain.volo_view();
auto vnew = domain.vnew_view();

auto delx_zeta = domain.delx_zeta_view();
auto delx_xi   = domain.delx_xi_view();
auto delx_eta  = domain.delx_eta_view();

auto delv_zeta = domain.delv_zeta_view();
auto delv_xi   = domain.delv_xi_view();
auto delv_eta  = domain.delv_eta_view();

// --- connectivity
auto nodelist = domain.nodelist_view();


    Kokkos::parallel_for(
        "CalcMonotonicQGradientsForElems",
        Kokkos::RangePolicy<>(0, numElem),
        KOKKOS_LAMBDA(const Index_t i)
    {
        const Real_t ptiny = Real_t(1.e-36);
        Real_t ax, ay, az;
        Real_t dxv, dyv, dzv;

        /* 
           Récupérer la connectivité élément→nœuds
           获取该单元的节点编号
        */

Index_t n0 = nodelist(8*i + 0);
Index_t n1 = nodelist(8*i + 1);
Index_t n2 = nodelist(8*i + 2);
Index_t n3 = nodelist(8*i + 3);
Index_t n4 = nodelist(8*i + 4);
Index_t n5 = nodelist(8*i + 5);
Index_t n6 = nodelist(8*i + 6);
Index_t n7 = nodelist(8*i + 7);

        /* 
           Charger les coordonnées (x,y,z)
           加载节点坐标
        */
        Real_t x0 = x(n0); Real_t x1 = x(n1);
        Real_t x2 = x(n2); Real_t x3 = x(n3);
        Real_t x4 = x(n4); Real_t x5 = x(n5);
        Real_t x6 = x(n6); Real_t x7 = x(n7);

        Real_t y0 = y(n0); Real_t y1 = y(n1);
        Real_t y2 = y(n2); Real_t y3 = y(n3);
        Real_t y4 = y(n4); Real_t y5 = y(n5);
        Real_t y6 = y(n6); Real_t y7 = y(n7);

        Real_t z0 = z(n0); Real_t z1 = z(n1);
        Real_t z2 = z(n2); Real_t z3 = z(n3);
        Real_t z4 = z(n4); Real_t z5 = z(n5);
        Real_t z6 = z(n6); Real_t z7 = z(n7);

        /* 
           Charger les vitesses nodales
           加载节点速度
        */
        Real_t xv0 = xd(n0); Real_t xv1 = xd(n1);
        Real_t xv2 = xd(n2); Real_t xv3 = xd(n3);
        Real_t xv4 = xd(n4); Real_t xv5 = xd(n5);
        Real_t xv6 = xd(n6); Real_t xv7 = xd(n7);

        Real_t yv0 = yd(n0); Real_t yv1 = yd(n1);
        Real_t yv2 = yd(n2); Real_t yv3 = yd(n3);
        Real_t yv4 = yd(n4); Real_t yv5 = yd(n5);
        Real_t yv6 = yd(n6); Real_t yv7 = yd(n7);

        Real_t zv0 = zd(n0); Real_t zv1 = zd(n1);
        Real_t zv2 = zd(n2); Real_t zv3 = zd(n3);
        Real_t zv4 = zd(n4); Real_t zv5 = zd(n5);
        Real_t zv6 = zd(n6); Real_t zv7 = zd(n7);

        /* 
           Volume actuel = volo * vnew
           当前体积
        */
        Real_t vol = domain.volo(i) * domain.vnew(i);
        Real_t norm = Real_t(1.0) / (vol + ptiny);

        /*-----------------------------------------------*
         |                Direction  ζ                  |
         *-----------------------------------------------*/

        Real_t dxj = Real_t(-0.25) * ((x0+x1+x5+x4) - (x3+x2+x6+x7));
        Real_t dyj = Real_t(-0.25) * ((y0+y1+y5+y4) - (y3+y2+y6+y7));
        Real_t dzj = Real_t(-0.25) * ((z0+z1+z5+z4) - (z3+z2+z6+z7));

        Real_t dxi = Real_t(0.25) * ((x1+x2+x6+x5) - (x0+x3+x7+x4));
        Real_t dyi = Real_t(0.25) * ((y1+y2+y6+y5) - (y0+y3+y7+y4));
        Real_t dzi = Real_t(0.25) * ((z1+x2+z6+z5) - (z0+z3+z7+z4));

        Real_t dxk = Real_t(0.25) * ((x4+x5+x6+x7) - (x0+x1+x2+x3));
        Real_t dyk = Real_t(0.25) * ((y4+y5+y6+y7) - (y0+y1+y2+y3));
        Real_t dzk = Real_t(0.25) * ((z4+z5+z6+z7) - (z0+z1+z2+z3));

        /* (i cross j) */
        ax = dyi*dzj - dzi*dyj;
        ay = dzi*dxj - dxi*dzj;
        az = dxi*dyj - dyi*dxj;

        delx_zeta(i) = vol / sqrt(ax*ax + ay*ay + az*az + ptiny);

        ax *= norm; ay *= norm; az *= norm;

        dxv = Real_t(0.25) * ((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3));
        dyv = Real_t(0.25) * ((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3));
        dzv = Real_t(0.25) * ((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3));

        delv_zeta(i) = ax*dxv + ay*dyv + az*dzv;

        /*-----------------------------------------------*
         |                Direction  ξ                  |
         *-----------------------------------------------*/

        ax = dyj*dzk - dzj*dyk;
        ay = dzj*dxk - dxj*dzk;
        az = dxj*dyk - dyj*dxk;

        delx_xi(i) = vol / sqrt(ax*ax + ay*ay + az*az + ptiny);

        ax *= norm; ay *= norm; az *= norm;

        dxv = Real_t(0.25) * ((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4));
        dyv = Real_t(0.25) * ((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4));
        dzv = Real_t(0.25) * ((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4));

        delv_xi(i) = ax*dxv + ay*dyv + az*dzv;

        /*-----------------------------------------------*
         |                Direction  η                  |
         *-----------------------------------------------*/

        ax = dyk*dzi - dzk*dyi;
        ay = dzk*dxi - dxk*dzi;
        az = dxk*dyi - dyk*dxi;

        delx_eta(i) = vol / sqrt(ax*ax + ay*ay + az*az + ptiny);

        ax *= norm; ay *= norm; az *= norm;

        dxv = Real_t(-0.25) * ((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7));
        dyv = Real_t(-0.25) * ((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7));
        dzv = Real_t(-0.25) * ((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7));

        delv_eta(i) = ax*dxv + ay*dyv + az*dzv;
    });
}

/* 
   Calcule le q monotone pour une région d’éléments.
   对一个区域内的所有单元计算单调人工粘性 Q。
*/
KOKKOS_INLINE_FUNCTION
void CalcMonotonicQRegionForElems(Domain &domain, Int_t r, Real_t ptiny)
{
    Real_t monoq_limiter_mult = domain.monoq_limiter_mult();
    Real_t monoq_max_slope    = domain.monoq_max_slope();
    Real_t qlc_monoq          = domain.qlc_monoq();
    Real_t qqc_monoq          = domain.qqc_monoq();

    Index_t regSize = domain.regElemSize(r);

// -------- Region / topology --------
auto regElemlist = domain.regElemlist_view();
auto elemBC      = domain.elemBC_view();

// -------- Neighbors --------
auto lxim   = domain.lxim_view();
auto lxip   = domain.lxip_view();
auto letam  = domain.letam_view();
auto letap  = domain.letap_view();
auto lzetam = domain.lzetam_view();
auto lzetap = domain.lzetap_view();

// -------- Gradients --------
auto delv_xi   = domain.delv_xi_view();
auto delv_eta  = domain.delv_eta_view();
auto delv_zeta = domain.delv_zeta_view();

auto delx_xi   = domain.delx_xi_view();
auto delx_eta  = domain.delx_eta_view();
auto delx_zeta = domain.delx_zeta_view();

// -------- State --------
auto vdov     = domain.vdov_view();

auto volo     = domain.volo_view();
auto vnew     = domain.vnew_view();


auto ql = domain.ql_view();
auto qq = domain.qq_view();



    Kokkos::parallel_for(
        "CalcMonotonicQRegionForElems",
        Kokkos::RangePolicy<>(0, regSize),
        KOKKOS_LAMBDA(const Index_t idx)
    {
        Index_t ielem = regElemlist(r, idx);

        Real_t qlin, qquad;
        Real_t phixi, phieta, phizeta;
        Int_t bcMask = elemBC(ielem);
        Real_t delvm = 0.0, delvp = 0.0;

        /*---------------------------------------------------*
         |                  Direction  ξ                    |
         *---------------------------------------------------*/

        /* 
           norm = 1 / (delv_xi + ptiny)
           归一化系数（防止除零）
        */
        Real_t norm = Real_t(1.0) / (delv_xi(ielem) + ptiny);

        /* 
           BC du côté -ξ
           -ξ 边界条件处理
        */
        switch (bcMask & XI_M) {
            case XI_M_COMM: /* nécessite des données MPI（被禁用但逻辑保留） */
            case 0:
                delvm = delv_xi(lxim(ielem));
                break;
            case XI_M_SYMM:
                delvm = delv_xi(ielem);
                break;
            case XI_M_FREE:
                delvm = Real_t(0.0);
                break;
            default:
                delvm = Real_t(0.0);
                break;
        }

        /* BC du côté +ξ (+ξ 方向) */
        switch (bcMask & XI_P) {
            case XI_P_COMM:
            case 0:
                delvp = delv_xi(lxip(ielem));
                break;
            case XI_P_SYMM:
                delvp = delv_xi(ielem);
                break;
            case XI_P_FREE:
                delvp = Real_t(0.0);
                break;
            default:
                delvp = Real_t(0.0);
                break;
        }

        delvm *= norm;
        delvp *= norm;

        phixi = Real_t(.5) * (delvm + delvp);

        delvm *= monoq_limiter_mult;
        delvp *= monoq_limiter_mult;

        if (delvm < phixi) phixi = delvm;
        if (delvp < phixi) phixi = delvp;
        if (phixi < Real_t(0.0)) phixi = Real_t(0.0);
        if (phixi > monoq_max_slope) phixi = monoq_max_slope;

        /*---------------------------------------------------*
         |                 Direction  η                     |
         *---------------------------------------------------*/

        norm = Real_t(1.0) / (delv_eta(ielem) + ptiny);

        switch (bcMask & ETA_M) {
            case ETA_M_COMM:
            case 0:
                delvm = delv_eta(letam(ielem));
                break;
            case ETA_M_SYMM:
                delvm = delv_eta(ielem);
                break;
            case ETA_M_FREE:
                delvm = Real_t(0.0);
                break;
            default:
                delvm = Real_t(0.0);
        }

        switch (bcMask & ETA_P) {
            case ETA_P_COMM:
            case 0:
                delvp = delv_eta(letap(ielem));
                break;
            case ETA_P_SYMM:
                delvp = delv_eta(ielem);
                break;
            case ETA_P_FREE:
                delvp = Real_t(0.0);
                break;
            default:
                delvp = Real_t(0.0);
        }

        delvm *= norm;
        delvp *= norm;

        phieta = Real_t(.5) * (delvm + delvp);

        delvm *= monoq_limiter_mult;
        delvp *= monoq_limiter_mult;

        if (delvm < phieta) phieta = delvm;
        if (delvp < phieta) phieta = delvp;
        if (phieta < Real_t(0.0)) phieta = Real_t(0.0);
        if (phieta > monoq_max_slope) phieta = monoq_max_slope;

        /*---------------------------------------------------*
         |                 Direction  ζ                     |
         *---------------------------------------------------*/

        norm = Real_t(1.0) / (delv_zeta(ielem) + ptiny);

        switch (bcMask & ZETA_M) {
            case ZETA_M_COMM:
            case 0:
                delvm = delv_zeta(lzetam(ielem));
                break;
            case ZETA_M_SYMM:
                delvm = delv_zeta(ielem);
                break;
            case ZETA_M_FREE:
                delvm = Real_t(0.0);
                break;
            default:
                delvm = Real_t(0.0);
        }

        switch (bcMask & ZETA_P) {
            case ZETA_P_COMM:
            case 0:
                delvp = delv_zeta(lzetap(ielem));
                break;
            case ZETA_P_SYMM:
                delvp = delv_zeta(ielem);
                break;
            case ZETA_P_FREE:
                delvp = Real_t(0.0);
                break;
            default:
                delvp = Real_t(0.0);
        }

        delvm *= norm;
        delvp *= norm;

        phizeta = Real_t(.5) * (delvm + delvp);

        delvm *= monoq_limiter_mult;
        delvp *= monoq_limiter_mult;

        if (delvm < phizeta) phizeta = delvm;
        if (delvp < phizeta) phizeta = delvp;
        if (phizeta < Real_t(0.0)) phizeta = Real_t(0.0);
        if (phizeta > monoq_max_slope) phizeta = monoq_max_slope;

        /*---------------------------------------------------*
         |               Calcul final qlin / qquad          |
         *---------------------------------------------------*/

        if (vdov(ielem) > Real_t(0.0)) {
            qlin = Real_t(0.0);
            qquad = Real_t(0.0);
        }
        else {
            Real_t delv_xi_s   = delv_xi(ielem)   * delx_xi(ielem);
            Real_t delv_eta_s  = delv_eta(ielem)  * delx_eta(ielem);
            Real_t delv_zeta_s = delv_zeta(ielem) * delx_zeta(ielem);

            if (delv_xi_s   > Real_t(0.0)) delv_xi_s   = Real_t(0.0);
            if (delv_eta_s  > Real_t(0.0)) delv_eta_s  = Real_t(0.0);
            if (delv_zeta_s > Real_t(0.0)) delv_zeta_s = Real_t(0.0);

            Real_t rho = domain.elemMass(ielem) /
                         (volo(ielem) * vnew(ielem));

            qlin = -qlc_monoq * rho * (
                delv_xi_s   * (Real_t(1.0) - phixi) +
                delv_eta_s  * (Real_t(1.0) - phieta) +
                delv_zeta_s * (Real_t(1.0) - phizeta)
            );

            qquad = qqc_monoq * rho * (
                delv_xi_s*delv_xi_s     * (Real_t(1.0) - phixi*phixi) +
                delv_eta_s*delv_eta_s   * (Real_t(1.0) - phieta*phieta) +
                delv_zeta_s*delv_zeta_s * (Real_t(1.0) - phizeta*phizeta)
            );
        }

        ql(ielem) = qlin;
        qq(ielem) = qquad;
    });
}

/*
   Calcule l’artificial viscosity monotone pour tous les éléments,
   région par région.
   对所有区域逐个计算单调人工粘性 Q。
*/
KOKKOS_INLINE_FUNCTION
void CalcMonotonicQForElems(Domain& domain)
{
    /* 
       ptiny est une petite valeur pour éviter la division par zéro.
       ptiny 是用于避免除零的小量。
    */
    const Real_t ptiny = Real_t(1.e-36);

    /* 
       Parcourt toutes les régions.
       遍历所有区域。
    */
    for (Index_t r = 0; r < domain.numReg(); ++r) {

        /* 
           Si la région contient au moins un élément，则进行计算。
           如果区域内包含至少一个单元，则进行计算。
        */
        if (domain.regElemSize(r) > 0) {

            /* 
               Appel device-ready : fonctionne en CPU ou GPU。
               设备可调用版本：可在 CPU/GPU 运行。
            */
            CalcMonotonicQRegionForElems(domain, r, ptiny);
        }
    }
}

/*
   Calcule la viscosité artificielle Q pour tous les éléments.
   对所有单元计算人工粘性 Q。
*/
KOKKOS_INLINE_FUNCTION
void CalcQForElems(Domain& domain)
{
    /* 
       Nombre total d’éléments locaux.
       本地单元总数。
    */
    Index_t numElem = domain.numElem();

    if (numElem == 0) {
        return;
    }

    /*
       allElem = nombre total incluant les ghost layers。
       allElem = 含幽灵层在内的全部元素数量。
      （在 MPI=0 时，domain.AllocateGradients 仍需要此尺寸参数）
    */
    Int_t allElem =
          numElem
        + 2 * domain.sizeX() * domain.sizeY()
        + 2 * domain.sizeX() * domain.sizeZ()
        + 2 * domain.sizeY() * domain.sizeZ();

    /*
       Alloue les gradients nécessaires pour delv_xi, delv_eta, delv_zeta.
       为 delv_xi / delv_eta / delv_zeta 分配存储。
    */
    domain.AllocateGradients(numElem, allElem);

    /*
       Calcule les gradients de vitesse。
       计算速度梯度。
    */
    CalcMonotonicQGradientsForElems(domain);

    /*
       Calcule Q monotone pour chaque région。
       对每个区域计算单调人工粘性 Q。
    */
    {
        const Real_t ptiny = Real_t(1.e-36);

        for (Index_t r = 0; r < domain.numReg(); ++r) {
            if (domain.regElemSize(r) > 0) {
                CalcMonotonicQRegionForElems(domain, r, ptiny);
            }
        }
    }

    /*
       Libère les données de gradient。
       释放梯度内存。
    */
    domain.DeallocateGradients();

    /*
       Vérifie si un élément dépasse la limite QStop。
       检查是否有单元超过 QStop 阈值。
    */
    Index_t idx = -1;

    for (Index_t i = 0; i < numElem; ++i) {
        if (domain.q(i) > domain.qstop()) {
            idx = i;
            break;
        }
    }

    /* 
       Si Q dépasse la limite, arrêter le programme（MPI=0）。
       如果 Q 超过限制，则退出程序（MPI=0）。
    */
    if (idx >= 0) {
        exit(QStopError);
    }
}

/*
   Calcule la pression pour chaque élément d'une région.
   根据区域内的元素计算压力（p_new）。
*/
KOKKOS_INLINE_FUNCTION
void CalcPressureForElems(Real_t* p_new,
                          Real_t* bvc,
                          Real_t* pbvc,
                          Real_t* e_old,
                          Real_t* compression,
                          Real_t* vnewc,
                          Real_t  pmin,
                          Real_t  p_cut,
                          Real_t  eosvmax,
                          Index_t length,
                          Index_t* regElemList)
{
    /* 
       Première boucle : calcul de bvc[] et pbvc[]
       第一阶段：计算 bvc 与 pbvc
    */
    Kokkos::parallel_for(
        "CalcPressureForElems_stage1",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Real_t c1s = Real_t(2.0) / Real_t(3.0);
        bvc[i]  = c1s * (compression[i] + Real_t(1.0));
        pbvc[i] = c1s;
    });

    /* 
       Deuxième boucle : calcule p_new[i] à partir de e_old et contraintes EOS.
       第二阶段：根据旧能量 e_old 和 EOS 限制计算新的压力 p_new。
    */
    Kokkos::parallel_for(
        "CalcPressureForElems_stage2",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Index_t ielem = regElemList[i];

        // p_new = bvc * e_old
        // 基础压力公式：p_new = bvc * e_old
        Real_t ptmp = bvc[i] * e_old[i];
        p_new[i] = ptmp;

        // |p| < p_cut → p = 0
        // 绝对值太小则置为 0
        if (FABS(p_new[i]) < p_cut) {
            p_new[i] = Real_t(0.0);
        }

        // 如果体积超过上限（通常不会触发）
        if (vnewc[ielem] >= eosvmax) {
            p_new[i] = Real_t(0.0);
        }

        // 限制最低压力
        if (p_new[i] < pmin) {
            p_new[i] = pmin;
        }
    });
}

/*
   Met à jour l'énergie interne, la pression et la viscosité artificielle q
   更新单元的内部能量、压力 p_new、人工粘性 q_new
*/
KOKKOS_INLINE_FUNCTION
void CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old, Real_t* e_old, Real_t* q_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t* delvc,
                        Real_t pmin, Real_t p_cut, Real_t e_cut,
                        Real_t q_cut, Real_t emin,
                        Real_t* qq_old, Real_t* ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length, Index_t* regElemList)
{
    /* 
       Tableau temporaire pHalfStep utilisé par l'EOS.
       EOS 中间步骤压力 pHalfStep。
    */
    Real_t* pHalfStep = Allocate<Real_t>(length);

    /*--------------------------------------------------------------*
     |  Étape 1 : calcul initial de e_new                           |
     |  第 1 阶段：计算内部能量 e_new（初步更新）                     |
     *--------------------------------------------------------------*/
    Kokkos::parallel_for(
        "CalcEnergyForElems_e_new_stage1",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        // e_new = e_old - 1/2 * delvc * (p_old + q_old) + 1/2 * work
        // 第一次能量更新公式
        Real_t en = e_old[i]
            - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
            + Real_t(0.5) * work[i];

        // Clip to minimum
        // 限制不能低于 emin
        if (en < emin) {
            en = emin;
        }

        e_new[i] = en;
    });

    /*--------------------------------------------------------------*
     |  Étape 2 : calcul de pression pHalfStep                      |
     |  第 2 阶段：用初始 e_new 计算第一次压力 pHalfStep             |
     *--------------------------------------------------------------*/
    CalcPressureForElems(
        pHalfStep,
        bvc, pbvc,
        e_new,
        compHalfStep,
        vnewc,
        pmin, p_cut, eosvmax,
        length,
        regElemList
    );

        /*--------------------------------------------------------------*
     |  Étape 3 : calcul de q_new (1ère version)                    |
     |  第 3 阶段：计算一次更新后的 q_new                           |
     *--------------------------------------------------------------*/
    Kokkos::parallel_for(
        "CalcEnergyForElems_q_new_stage1",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Index_t ielem = regElemList[i];
        Real_t vhalf = Real_t(1.0) / (Real_t(1.0) + compHalfStep[i]);

        if (delvc[i] > Real_t(0.0)) {
            // Expansion → no viscous term
            // 膨胀过程 → 人工粘性为 0
            q_new[i] = Real_t(0.0);
        }
        else {
            // ssc = speed of sound squared (scaled)
            // ssc = 声速相关参数
            Real_t ssc =
                  ( pbvc[i] * e_new[i]
                  + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0;

            if (ssc <= Real_t(1.111111e-36)) {
                ssc = Real_t(3.333333e-18);
            } else {
                ssc = SQRT(ssc);
            }

            // q_new = ssc * ql_old + qq_old
            // 线性+二次人工粘性组合
            q_new[i] = ssc * ql_old[i] + qq_old[i];
        }

        // 第二次能量更新准备
        // e_new = e_new + 0.5 * delvc * (...)
        // 具体表达式下一段完成
    });

    /*--------------------------------------------------------------*
     |  Étape 4 : deuxième mise à jour de e_new                      |
     |  第 4 阶段：对 e_new 进行第二次更新                           |
     *--------------------------------------------------------------*/
    Kokkos::parallel_for(
        "CalcEnergyForElems_e_new_stage2",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Index_t ielem = regElemList[i];

        // e_new += 0.5 * delvc * (3*(p_old+q_old) - 4*(pHalfStep+q_new))
        // 内部能量第二次修正
        e_new[i] = e_new[i] +
            Real_t(0.5) * delvc[i] *
            ( Real_t(3.0) * (p_old[i] + q_old[i])
            - Real_t(4.0) * (pHalfStep[i] + q_new[i]) );
    });

    /*--------------------------------------------------------------*
     |  Étape 5 : corriger e_new et recalculer p avec EOS           |
     |  第 5 阶段：根据阈值修正 e_new，并重新计算压力 p_new          |
     *--------------------------------------------------------------*/
    Kokkos::parallel_for(
        "CalcEnergyForElems_e_new_stage3_cleanup",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        // Ajoute travail résiduel
        // 加上剩余 work
        e_new[i] += Real_t(0.5) * work[i];

        // |e| < e_cut → 0
        if (FABS(e_new[i]) < e_cut)
            e_new[i] = Real_t(0.0);

        // e < emin → clip
        if (e_new[i] < emin)
            e_new[i] = emin;
    });

    /*--------------------------------------------------------------*
     |  Étape 6 : recalcul de p_new après correction d'énergie      |
     |  第 6 阶段：能量修正后重新计算压力 p_new                      |
     *--------------------------------------------------------------*/
    CalcPressureForElems(
        p_new,
        bvc, pbvc,
        e_new,
        compression,
        vnewc,
        pmin, p_cut, eosvmax,
        length,
        regElemList
    );

        /*--------------------------------------------------------------*
     |  Étape 7 : calcul final de q_tilde                            |
     |  第 7 阶段：计算最终人工粘性 q_tilde                         |
     *--------------------------------------------------------------*/
    Kokkos::parallel_for(
        "CalcEnergyForElems_q_tilde_final",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        const Real_t sixth = Real_t(1.0) / Real_t(6.0);
        Index_t ielem = regElemList[i];
        Real_t q_tilde;

        if (delvc[i] > Real_t(0.0)) {
            // Expansion → no viscosity
            // 膨胀过程 → 人工粘性为 0
            q_tilde = Real_t(0.0);
        }
        else {
            // Speed-of-sound-like term
            // 声速相关参数
            Real_t ssc =
                ( pbvc[i] * e_new[i]
                + vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i] ) / rho0;

            if (ssc <= Real_t(1.111111e-36)) {
                ssc = Real_t(3.333333e-18);
            }
            else {
                ssc = SQRT(ssc);
            }

            // Final viscous term
            // 最终人工粘性
            q_tilde = ssc * ql_old[i] + qq_old[i];
        }

        // e_new = e_new − (7*(p_old+q_old) − 8*(pHalfStep+q_new) + (p_new+q_tilde)) * delvc/6
        // e_new 最终修正
        e_new[i] =
            e_new[i]
            - (  Real_t(7.0) * (p_old[i] + q_old[i])
               - Real_t(8.0) * (pHalfStep[i] + q_new[i])
               + (p_new[i] + q_tilde) )
              * delvc[i] * sixth;

        // Nettoyage basé sur e_cut et emin
        // 根据 e_cut 和 emin 清理
        if (FABS(e_new[i]) < e_cut) {
            e_new[i] = Real_t(0.0);
        }
        if (e_new[i] < emin) {
            e_new[i] = emin;
        }
    });

    /*--------------------------------------------------------------*
     |  Étape 8 : recalcul final de p_new et limitation de q_new    |
     |  第 8 阶段：最终计算压力 p_new，并对 q_new 应用阈值限制       |
     *--------------------------------------------------------------*/
    CalcPressureForElems(
        p_new,
        bvc, pbvc,
        e_new,
        compression,
        vnewc,
        pmin, p_cut, eosvmax,
        length,
        regElemList
    );

    Kokkos::parallel_for(
        "CalcEnergyForElems_q_new_clamp",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Index_t ielem = regElemList[i];

        if (delvc[i] <= Real_t(0.0)) {
            // Compute ssc again
            Real_t ssc =
                ( pbvc[i] * e_new[i]
                + vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i] ) / rho0;

            if (ssc <= Real_t(1.111111e-36)) {
                ssc = Real_t(3.333333e-18);
            }
            else {
                ssc = SQRT(ssc);
            }

            q_new[i] = ssc * ql_old[i] + qq_old[i];

            // Apply q_cut
            // 应用人工粘性阈值
            if (FABS(q_new[i]) < q_cut) {
                q_new[i] = Real_t(0.0);
            }
        }
    });

    /*--------------------------------------------------------------*
     |  Étape 9 : libération des tableaux temporaires               |
     |  第 9 阶段：释放所有临时数组                                 |
     *--------------------------------------------------------------*/
    Release(&pHalfStep);
    // NOTE：下列数组由外层 EvalEOSForElems 释放，而非此函数
    // 注意：其他临时数组由 EvalEOSForElems 统一管理，不应在此释放
}

/*
   Évalue l’équation d’état (EOS) pour une région donnée.
   为指定区域计算状态方程（EOS）。
   （GPU-ready 版本：所有临时数组均使用 Kokkos::View）
*/


// ------------------------------------------------------------
// Sound speed computation (Kokkos5-friendly)
// - regElemList: element ids (device pointer OK; we pass devElemList.data())
// - enewc/pnewc/pbvc/bvc: region-local arrays indexed by i in [0,numElemReg)
// - vnewc: global array indexed by element id
// ------------------------------------------------------------
void CalcSoundSpeedForElems(
    Domain& domain,
    const Real_t* vnewc,
    const Real_t rho0,
    const Real_t* enewc,
    const Real_t* pnewc,
    const Real_t* pbvc,
    const Real_t* bvc,
    const Real_t ss4o3,           // kept for signature compatibility
    const Index_t numElemReg,
    const Index_t* regElemList
)
{
    (void)ss4o3;
    if (numElemReg <= 0) return;

    auto ss = domain.ss_view(); // View<Real_t*>

    Kokkos::parallel_for(
        "CalcSoundSpeedForElems",
        Kokkos::RangePolicy<>(0, numElemReg),
        KOKKOS_LAMBDA(const Index_t i)
        {
            const Index_t ielem = regElemList[i];
            const Real_t  v     = vnewc[ielem];

            // Reference LULESH formula:
            // ssc = (pbvc*e + v^2*bvc*p) / rho0
            Real_t ssc = (pbvc[i] * enewc[i] + (v * v) * bvc[i] * pnewc[i]) / rho0;

            // Clamp like reference implementation
            if (ssc <= Real_t(1.111111e-36)) {
                ssc = Real_t(3.333333e-18);
            } else {
                ssc = SQRT(ssc);
            }

            ss(ielem) = ssc;
        }
    );
}

void EvalEOSForElems(
    Domain& domain,
    Real_t *vnewc,                // 已在调用方分配的 Kokkos::View.data()
    Int_t numElemReg,             // 该区域含多少单元
    Index_t *regElemList,         // 该区域的单元编号列表（由 Domain 提供）
    Int_t rep                     // 重复次数，用于 workload imbalance
)
{
    /* --- paramètres EOS --- */
    Real_t e_cut   = domain.e_cut();
    Real_t p_cut   = domain.p_cut();
    Real_t ss4o3   = domain.ss4o3();
    Real_t q_cut   = domain.q_cut();

    Real_t eosvmax = domain.eosvmax();
    Real_t eosvmin = domain.eosvmin();
    Real_t pmin    = domain.pmin();
    Real_t emin    = domain.emin();
    Real_t rho0    = domain.refdens();


    /* ---------------------------------------------
       创建所有临时数组（device-ready）
       所有数组大小为 numElemReg
       --------------------------------------------- */

    Kokkos::View<Real_t*> e_old("e_old", numElemReg);
    Kokkos::View<Real_t*> delvc("delvc", numElemReg);
    Kokkos::View<Real_t*> p_old("p_old", numElemReg);
    Kokkos::View<Real_t*> q_old("q_old", numElemReg);
    Kokkos::View<Real_t*> compression("compression", numElemReg);
    Kokkos::View<Real_t*> compHalfStep("compHalfStep", numElemReg);
    Kokkos::View<Real_t*> qq_old("qq_old", numElemReg);
    Kokkos::View<Real_t*> ql_old("ql_old", numElemReg);
    Kokkos::View<Real_t*> work("work", numElemReg);
    Kokkos::View<Real_t*> p_new("p_new", numElemReg);
    Kokkos::View<Real_t*> e_new("e_new", numElemReg);
    Kokkos::View<Real_t*> q_new("q_new", numElemReg);
    Kokkos::View<Real_t*> bvc("bvc", numElemReg);
    Kokkos::View<Real_t*> pbvc("pbvc", numElemReg);

    /* 
       因为 regElemList 是 host pointer，
       我们复制到 device View 以便 Kokkos lambda 使用。
       因此 GPU 完全可用。
    */

    Kokkos::View<Index_t*> devElemList("devElemList", numElemReg);
    {
        auto host = Kokkos::create_mirror_view(devElemList);
        for (Int_t i = 0; i < numElemReg; i++) host(i) = regElemList[i];
        Kokkos::deep_copy(devElemList, host);
    }

    /* ----------------------------------------------------
       主循环：重复 rep 次（用于 load imbalance）
       每次重复都执行完整 EOS 更新流程
       ---------------------------------------------------- */
    for (Int_t j = 0; j < rep; j++) {

        /* 
           STEP 1:
           压缩 e_old / delvc / p_old / q_old / qq_old / ql_old
           法语：charger les anciennes valeurs des éléments。
        */


auto e    = domain.e_view();
auto delv = domain.delv_view();
auto p    = domain.p_view();
auto q    = domain.q_view();
auto qq   = domain.qq_view();
auto ql   = domain.ql_view();


        Kokkos::parallel_for(
            "EOS_load_old",
            Kokkos::RangePolicy<>(0, numElemReg),
            KOKKOS_LAMBDA(const Index_t i)
        {
            Index_t ielem = devElemList(i);

            e_old(i)  = e(ielem);
            delvc(i)  = delv(ielem);
            p_old(i)  = p(ielem);
            q_old(i)  = q(ielem);
            qq_old(i) = qq(ielem);
            ql_old(i) = ql(ielem);
        });

        /* 
           STEP 2:
           计算 compression[i] = 1/vnew - 1
           计算 compHalfStep[i] = 1/v_half - 1
           法语：calculer la compression et demie-compression。
        */
        Kokkos::parallel_for(
            "EOS_compression",
            Kokkos::RangePolicy<>(0, numElemReg),
            KOKKOS_LAMBDA(const Index_t i)
        {
            Index_t ielem = devElemList(i);

            Real_t vchalf = vnewc[ielem] - delvc(i) * Real_t(0.5);
            compression(i)   = Real_t(1.0)/vnewc[ielem] - Real_t(1.0);
            compHalfStep(i)  = Real_t(1.0)/vchalf       - Real_t(1.0);
        });

        /* 
           STEP 3:
           边界检查：处理 eosvmin / eosvmax 对压缩量的裁剪
           法语：contrôle des volumes limites。
        */
        if (eosvmin != Real_t(0.0)) {
            Kokkos::parallel_for(
                "EOS_eosvmin_check",
                Kokkos::RangePolicy<>(0, numElemReg),
                KOKKOS_LAMBDA(const Index_t i)
            {
                Index_t ielem = devElemList(i);
                if (vnewc[ielem] <= eosvmin) {
                    compHalfStep(i) = compression(i);
                }
            });
        }

        if (eosvmax != Real_t(0.0)) {
            Kokkos::parallel_for(
                "EOS_eosvmax_check",
                Kokkos::RangePolicy<>(0, numElemReg),
                KOKKOS_LAMBDA(const Index_t i)
            {
                Index_t ielem = devElemList(i);
                if (vnewc[ielem] >= eosvmax) {
                    p_old(i)        = Real_t(0.0);
                    compression(i)  = Real_t(0.0);
                    compHalfStep(i) = Real_t(0.0);
                }
            });
        }

        /* 
           STEP 4:
           初始化 work = 0
           法语：initialiser work à zéro。
        */
        Kokkos::parallel_for(
            "EOS_reset_work",
            Kokkos::RangePolicy<>(0, numElemReg),
            KOKKOS_LAMBDA(const Index_t i)
        {
            work(i) = Real_t(0.0);
        });

        /* 
           STEP 5:
           调用 GPU-ready CalcEnergyForElems（上一部分已移植）
        */
        CalcEnergyForElems(
    p_new.data(), e_new.data(), q_new.data(),
    bvc.data(), pbvc.data(),
    p_old.data(), e_old.data(), q_old.data(),
    compression.data(), compHalfStep.data(),
    vnewc, work.data(), delvc.data(),
    pmin, p_cut, e_cut, q_cut, emin,
    qq_old.data(), ql_old.data(),
    rho0, eosvmax,
    numElemReg,
    devElemList.data()
);

    } // end for(rep)

auto p = domain.p_view();
auto e = domain.e_view();
auto q = domain.q_view();


    /* -------------------------------
       将 e_new / p_new / q_new 写回 Domain
       法语：mettre à jour le domaine。
       ------------------------------- */
    Kokkos::parallel_for(
        "EOS_writeback",
        Kokkos::RangePolicy<>(0, numElemReg),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Index_t ielem = devElemList(i);
        p(ielem) = p_new(i);
        e(ielem) = e_new(i);
        q(ielem) = q_new(i);
    });

    /* 
       STEP 7:
       计算 Sound Speed（调用已移植的 GPU-ready 函数）
    */
   CalcSoundSpeedForElems(
       domain,
       vnewc, rho0,
       e_new.data(), p_new.data(),
       pbvc.data(), bvc.data(),
       ss4o3,
       numElemReg, devElemList.data()
   );
}

/*
   Applique les propriétés matériaux aux éléments.
   为所有单元应用材料属性。
   （GPU-ready：所有循环迁移至 Kokkos::parallel_for）
*/
void ApplyMaterialPropertiesForElems(Domain& domain)
{
    Index_t numElem = domain.numElem();
    if (numElem == 0) return;


    /* ----------------------------------------------------------
       Préparer les volumes relatifs vnewc（新体积的副本）
       由于 vnewc 是临时量，因此在 device 上缓存 View
       ---------------------------------------------------------- */
    Kokkos::View<Real_t*> vnewc("vnewc", numElem);

    /* 
       Charger vnewc depuis domain.vnew(i)
       从 domain 复制 vnew(i) → vnewc(i)
    */
    Kokkos::parallel_for(
        "Load_vnewc",
        Kokkos::RangePolicy<>(0, numElem),
        KOKKOS_LAMBDA(const Index_t i)
    {
        vnewc(i) = domain.vnew(i);
    });

    /* 
       Récupérer les bornes EOS（eosvmin / eosvmax）
       读取 EOS 模型的体积限制
    */
    const Real_t eosvmin = domain.eosvmin();
    const Real_t eosvmax = domain.eosvmax();

    /* ----------------------------------------------------------
       Contraindre vnewc dans [eosvmin, eosvmax]
       将 vnewc 限制在 EOS 允许范围内。
       ---------------------------------------------------------- */
    if (eosvmin != Real_t(0.0)) {
        Kokkos::parallel_for(
            "Clamp_vnewc_min",
            Kokkos::RangePolicy<>(0, numElem),
            KOKKOS_LAMBDA(const Index_t i)
        {
            if (vnewc(i) < eosvmin)
                vnewc(i) = eosvmin;
        });
    }

    if (eosvmax != Real_t(0.0)) {
        Kokkos::parallel_for(
            "Clamp_vnewc_max",
            Kokkos::RangePolicy<>(0, numElem),
            KOKKOS_LAMBDA(const Index_t i)
        {
            if (vnewc(i) > eosvmax)
                vnewc(i) = eosvmax;
        });
    }

    /* ----------------------------------------------------------
       Vérifier les anciens volumes domain.v(i)
       （这在原代码中虽不完全合理，但保持一致性）
       检查旧体积 v(i) 是否在合理范围内
       ---------------------------------------------------------- */
    Kokkos::parallel_for(
        "Check_old_v",
        Kokkos::RangePolicy<>(0, numElem),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Real_t vc = domain.v(i);

        if (eosvmin != Real_t(0.0)) {
            if (vc < eosvmin)
                vc = eosvmin;
        }
        if (eosvmax != Real_t(0.0)) {
            if (vc > eosvmax)
                vc = eosvmax;
        }

        /* 
           Si volume < 0 → erreur fatale（MPI = 0）
           如果体积负 → 程序终止
        */
        if (vc <= 0.0) {
            exit(VolumeError);
        }
    });

    /* ----------------------------------------------------------
       Boucler sur chaque région r
       对每个区域调用 EvalEOSForElems（GPU-ready 版本）
       ---------------------------------------------------------- */
    for (Int_t r = 0; r < domain.numReg(); r++) {

        Index_t numElemReg = domain.regElemSize(r);
        if (numElemReg == 0) continue;
Kokkos::View<Index_t*> devElemList("devElemList", numElemReg);
{
    auto host = Kokkos::create_mirror_view(devElemList);
    auto regElemlist = domain.regElemlist_view();
    for (Index_t i = 0; i < numElemReg; ++i)
        host(i) = regElemlist(r, i);
    Kokkos::deep_copy(devElemList, host);
}


        

        /* 
           Déterminer le coût（workload imbalance simulation）
           决定该区域的 rep 次数（模拟负载不均衡）
        */
        Int_t rep;
        if (r < domain.numReg() / 2)
            rep = 1;
        else if (r < (domain.numReg() - (domain.numReg() + 15) / 20))
            rep = 1 + domain.cost();
        else
            rep = 10 * (1 + domain.cost());

        /* 
           Appeler la version GPU de EvalEOSForElems
           调用已移植的 GPU-ready 状态方程（EOS）计算函数
        */
        EvalEOSForElems(
            domain,
            vnewc.data(),     // device pointer
            numElemReg,
            devElemList.data(),      // host pointer → EvalEOS 内部会复制
            rep
        );
    }

    /* 
       vnewc 将在函数结束时自动释放
       无需手动 Release()
    */
}

/*
   Met à jour les volumes des éléments après l'étape Lagrangienne.
   在拉格朗日更新步骤后更新单元体积 v(i)。

   GPU-ready：使用 Kokkos::parallel_for。
*/
KOKKOS_INLINE_FUNCTION
void UpdateVolumesForElems(Domain &domain,
                           Real_t v_cut,   // seuil pour correction du volume
                           Index_t length) // nombre total d’éléments
{
    if (length == 0) return;
    auto v = domain.v_view();

    Kokkos::parallel_for(
        "UpdateVolumesForElems",
        Kokkos::RangePolicy<>(0, length),
        KOKKOS_LAMBDA(const Index_t i)
    {
        /* 
           tmpV = vnew(i)
           新体积读取
        */
        Real_t tmpV = domain.vnew(i);

        /* 
           Si |tmpV - 1.0| < v_cut → forcer tmpV = 1.0
           如果体积变化非常小，则直接设为 1.0
           （这是 LULESH 中用于稳定性的经验规则）
        */
        if (FABS(tmpV - Real_t(1.0)) < v_cut) {
            tmpV = Real_t(1.0);
        }

        /* 
           Écrit le volume final dans domain.v(i)
           写回最终体积
        */
        v(i) = tmpV;
    });
}

/*
   Avance la solution en appliquant les équations de Lagrange
   pour les éléments.
   对所有单元执行拉格朗日推进步骤（元素部分）。

   GPU-ready：内部调用的所有函数均已 Kokkos 并行化。
*/

KOKKOS_INLINE_FUNCTION
void LagrangeElements(Domain& domain, Index_t numElem)
{
    /*
       Étape 1 : Calcul Lagrangien élémentaire
       步骤 1：计算拉格朗日单元动力学（变形、应变率等）
    */
    CalcLagrangeElements(domain);

    /*
       Étape 2 : Calcul de la viscosité artificielle Q
       步骤 2：计算人工粘性 Q（单调性版本）
       注：内部调用 CalcMonotonicQGradientsForElems + CalcMonotonicQRegionForElems
    */
    CalcQForElems(domain);

    /*
       Étape 3 : Calcul des propriétés matériaux (EOS)
       步骤 3：计算材料状态（EOS：压力、能量、声速等）
       GPU-ready：EvalEOSForElems 已实现 Kokkos::View 完全并行
    */
    ApplyMaterialPropertiesForElems(domain);

    /*
       Étape 4 : Mise à jour des volumes v(i)
       步骤 4：更新体积 v(i)
       使用 GPU-ready 的 UpdateVolumesForElems
    */
    UpdateVolumesForElems(domain, 
                          domain.v_cut(),  // seuil de correction 体积修正阈值
                          numElem);        // nombre total 总单元数
}

/*
   Calcule la contrainte de Courant pour un ensemble d’éléments.
   计算 Courant 时间步约束（用于稳定性限制）。
*/
void CalcCourantConstraintForElems(
    Domain& domain,
    Index_t length,
    Kokkos::View<const Index_t*> regElemlist, /* region element list (device-accessible) */
    Real_t qqc2,
    Real_t& dtcourant_out)
{

  // IMPORTANT: treat incoming parameter as qqc, compute qqc2 exactly as reference:
  const Real_t qqc_in = (Real_t)(qqc2);
  const Real_t qqc2_local = Real_t(64.0) * qqc_in * qqc_in;

  Real_t dtcourant_reg = Real_t(1.0e20);

  Kokkos::parallel_reduce(
    "CalcCourantConstraintForElems",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, length),
    KOKKOS_LAMBDA(const Index_t i, Real_t& lmin) {
      const Index_t indx = regElemlist(i);

      const Real_t vdov = domain.vdov(indx);
      if (vdov != Real_t(0.0)) {
        const Real_t arealg = domain.arealg(indx);
        const Real_t ss     = domain.ss(indx);

        Real_t dtf = ss * ss;
        if (vdov < Real_t(0.0)) {
          const Real_t a2 = arealg * arealg;
          dtf += qqc2_local *  a2 * vdov * vdov;
        }
        dtf = arealg / Kokkos::sqrt(dtf);
        if (dtf < lmin) lmin = dtf;
      }
    },
    Kokkos::Min<Real_t>(dtcourant_reg)
  );

  dtcourant_out = dtcourant_reg;
}

/*
   Calcule la contrainte hydro-dynamique sur le pas de temps.
   计算流体力学时间步约束（基于体积变化率 vdov）。
*/
void CalcHydroConstraintForElems(
    Domain& domain,
    Index_t length,
    Kokkos::View<const Index_t*> regElemlist, /* region element list (device-accessible) */
    Real_t dvovmax,
    Real_t& dthydro_out)
{

  const Real_t dvovmax_in = (Real_t)(dvovmax);

  Real_t dthydro_reg = Real_t(1.0e20);

  Kokkos::parallel_reduce(
    "CalcHydroConstraintForElems",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, length),
    KOKKOS_LAMBDA(const Index_t i, Real_t& lmin) {
      const Index_t indx = regElemlist(i);

      const Real_t vdov = domain.vdov(indx);
      if (vdov != Real_t(0.0)) {
        const Real_t denom  = Kokkos::fabs(vdov) + Real_t(1.0e-20);
        const Real_t dtdvov = dvovmax_in / denom;
        if (dtdvov < lmin) lmin = dtdvov;
      }
    },
    Kokkos::Min<Real_t>(dthydro_reg)
  );

  dthydro_out = dthydro_reg;
}

/*
   Évalue les contraintes de temps (Courant et Hydro) pour tous les éléments.
   对全部区域（region）的单元计算时间步约束（Courant 与 Hydro）。
*/
void CalcTimeConstraintsForElems(Domain& domain)
{

  // Global init (do NOT overwrite per region)
  domain.dtcourant() = Real_t(1.0e20);
  domain.dthydro()   = Real_t(1.0e20);

  const Real_t qqc     = domain.qqc();
  const Real_t dvovmax = domain.dvovmax();

  const Index_t numReg = domain.numReg();
  auto regElemlist2d = domain.regElemlist_view();
  for (Index_t r = 0; r < numReg; ++r) {
    const Index_t length = domain.regElemSize(r);
    if (length <= 0) continue;

    auto regElemlist = Kokkos::subview(regElemlist2d, r, Kokkos::make_pair((Index_t)0, length));

    Real_t dtc_r = Real_t(1.0e20);
    Real_t dth_r = Real_t(1.0e20);

    CalcCourantConstraintForElems(domain, length, regElemlist, qqc, dtc_r);
    CalcHydroConstraintForElems  (domain, length, regElemlist, dvovmax, dth_r);

    if (dtc_r < domain.dtcourant()) domain.dtcourant() = dtc_r;
    if (dth_r < domain.dthydro())   domain.dthydro()   = dth_r;
  }
}

/*
   Effectue un pas de temps Lagrangien (méthode leap-frog).
   执行一次 Lagrange 弹跳法（Leapfrog）时间推进步骤。
*/
KOKKOS_INLINE_FUNCTION
void LagrangeLeapFrog(Domain& domain)
{
    /*
       Calcul des forces nodales, accélérations, vitesses et positions
       en tenant compte des conditions aux limites.
       计算节点力、加速度、速度、位置，并处理边界条件。
    */
    LagrangeNodal(domain);

    /*
       Calcul des quantités élémentaires (gradient de vitesse, viscosité q)
       puis mise à jour de l’état du matériau.
       计算单元量（速度梯度、人工粘性 q），并更新材料状态。
    */
    LagrangeElements(domain, domain.numElem());

    /*
       Évaluer les contraintes de temps (Courant + Hydro)
       计算时间步长约束（Courant + Hydro）
    */
    CalcTimeConstraintsForElems(domain);

    /*
       NOTE IMPORTANTE :
       Dans la version MPI, il existe des synchronisations de position/vitesse
       (SEDOV_SYNC_POS_VEL_LATE). Celles-ci ne sont pas utilisées ici
       car MPI = 0 dans la version Kokkos CPU/GPU.
       重要说明：
       在 MPI 版本中，这里会有位置/速度同步代码。
       本 GPU/Kokkos 单机版本保持禁用，因为 MPI = 0。
    */
}

/*
   Calcule l'incrément de temps (dt) pour l'itération suivante.
   根据 Courant/Hydro 约束计算下一步时间增量 dt。
*/
#if 0
KOKKOS_INLINE_FUNCTION
void TimeIncrement(Domain& domain)
{
    // Valeurs courantes de dt, temps, etc.
    // 当前 dt、时间等
    Real_t dtcourant = domain.dtcourant();
    Real_t dthydro   = domain.dthydro();

    Real_t olddt = domain.deltatime();
    Real_t newdt = olddt;

    /*
       Calcul du dt basé sur les contraintes physiques
       根据物理约束 (Courant + Hydro) 选择最小的 dt
    */
    if (dtcourant > Real_t(0.0)) {
        newdt = FMIN(newdt, dtcourant);
    }
    if (dthydro > Real_t(0.0)) {
        newdt = FMIN(newdt, dthydro);
    }

    /*
       Ajustement du dt si changement trop brutal
       如果 dt 变化过大，则进行平滑调整
       (对应 LULESH 原版的 dt "safety resize")
    */
    Real_t ratio = newdt / olddt;
    if (ratio >= Real_t(1.0)) {
        if (ratio < Real_t(1.2)) {
            // autorisé
            // 轻微变化允许
        } else {
            newdt = olddt * Real_t(1.2);
        }
    } else {
        if (ratio > Real_t(0.5)) {
            // autorisé
            // 轻微缩小允许
        } else {
            newdt = olddt * Real_t(0.5);
        }
    }

    /*
       Ne pas dépasser le temps final défini
       时间不能超过模拟终止点
    */
    Real_t stoptime = domain.stoptime();
    Real_t now      = domain.time();

    if ((now + newdt) > stoptime) {
        newdt = stoptime - now;
    }

    /*
       Mise à jour du domaine
       更新到 Domain 数据结构
    */
    domain.deltatimeold() = olddt;   // dt précédent / 旧 dt
    domain.deltatime()    = newdt;   // nouveau dt / 新 dt
    domain.time()        += newdt;   // avancer le temps / 时间推进

    domain.cycle()++;                // augmenter le compteur d’itérations
                                     // 迭代计数器 +1
}
#endif

/*
   Met à jour les quantités nodales : forces, accélérations, vitesses, positions.
   更新节点物理量：力、加速度、速度、位置。
*/
#if 0
KOKKOS_INLINE_FUNCTION
void LagrangeNodal(Domain& domain)
{
    /*
       1. Calcul des forces nodales
          计算节点力
    */
    CalcForceForNodes(domain);

    /*
       2. Calcul des accélérations nodales
          计算节点加速度 a = F / m
    */
    CalcAccelerationForNodes(domain);

    /*
       3. Application des conditions limites
          应用加速度的边界条件（固定面/对称面等）
    */
    ApplyAccelerationBoundaryConditionsForNodes(domain);

    /*
       4. Mise à jour des vitesses nodales
          更新节点速度：v(t+dt) = v(t) + a * dt
    */
    CalcVelocityForNodes(domain);

    /*
       5. Mise à jour des positions nodales
          更新节点位置：x(t+dt) = x(t) + v * dt
    */
    CalcPositionForNodes(domain);
}
#endif

#include <Kokkos_Core.hpp>

/*
   Programme principal de KLULES (version CPU sans MPI).
   KLULES 主程序（CPU 单机，无 MPI 版本）。
*/

int main(int argc, char* argv[])
{
    // Initialisation de Kokkos
    // 初始化 Kokkos
    Kokkos::initialize(argc, argv);
    {
        Domain* locDom ;
        struct cmdLineOpts opts;

        // --------------------------------------------------------------------
        // Définition des paramètres par défaut (modifiables par options CLI)
        // 设置默认参数（可被命令行覆盖）
        // --------------------------------------------------------------------
        opts.its      = 9999999;
        opts.nx       = 30;
        opts.numReg   = 11;
        opts.numFiles = 1;
        opts.showProg = 0;
        opts.quiet    = 0;
        opts.viz      = 0;
        opts.balance  = 1;
        opts.cost     = 1;

        // Analyse des options en ligne de commande
        // 解析命令行参数
        ParseCommandLineOptions(argc, argv, 0, &opts);

        if (!opts.quiet) {
            std::cout << "Running problem size "
                      << opts.nx << "^3 on single CPU domain\n";
            std::cout << "Total number of elements: "
                      << (opts.nx * opts.nx * opts.nx) << "\n\n";
        }

        // --------------------------------------------------------------------
        // Décomposition du maillage (version simple, pas de MPI)
        // 网格划分（单机版本，固定为 1 个域）
        // --------------------------------------------------------------------
        Int_t col = 0, row = 0, plane = 0, side = 1;
        InitMeshDecomp(1, 0, &col, &row, &plane, &side);

        // --------------------------------------------------------------------
        // Construction du domaine principal
        // 构造主 Domain 对象
        // --------------------------------------------------------------------
        locDom = new Domain(1, col, row, plane,
                            opts.nx, side,
                            opts.numReg, opts.balance, opts.cost);

        // --------------------------------------------------------------------
        // Boucle temporelle principale
        // 主时间推进循环
        // --------------------------------------------------------------------

        double start_time = wall_clock();   // 获取 wall-clock 时间

        while ((locDom->time() < locDom->stoptime()) &&
               (locDom->cycle() < opts.its))
        {
            // Incrément du temps
            // 时间步推进
            TimeIncrement(*locDom);

            // Mise à jour nodale + éléments
            // 节点更新 + 单元更新
            LagrangeLeapFrog(*locDom);

            if (opts.showProg && !opts.quiet) {
                std::cout << "cycle = " << locDom->cycle()
                          << ", time = " << double(locDom->time())
                          << ", dt = " << double(locDom->deltatime())
                          << "\n";
            }
        }

        double elapsed = wall_clock() - start_time;

        // --------------------------------------------------------------------
        // Sorties finales : fichier VTK / validation résultats
        // 最终输出：VTK 可视化 / 结果验证
        // --------------------------------------------------------------------
        if (opts.viz)
            DumpToVisit(*locDom, opts.numFiles, 0, 1);

        if (!opts.quiet)
            VerifyAndWriteFinalOutput(elapsed, *locDom, opts.nx, 1);

        delete locDom;
    }
    // Finalisation Kokkos
    // 结束 Kokkos
    Kokkos::finalize();

    return 0;
}

/* =========================================================
 * Array/pointer signature wrapper for CalcElemVolume
 * This matches the prototype in lulesh.h:
 *   Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8]);
 * ========================================================= */
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
  return CalcElemVolume(
    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
    y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
    z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]
  );
}

