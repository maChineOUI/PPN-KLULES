#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "lulesh.h"
#include <Kokkos_Core.hpp>
#ifdef VIZ_MESH

#ifdef __cplusplus
  extern "C" {
#endif
#include "silo.h"
#if USE_MPI
# include "pmpio.h"
#endif
#ifdef __cplusplus
  }
#endif

// Function prototypes
static void 
DumpDomainToVisit(DBfile *db, Domain& domain, int myRank);
static


#if USE_MPI
// For some reason, earlier versions of g++ (e.g. 4.2) won't let me
// put the 'static' qualifier on this prototype, even if it's done
// consistently in the prototype and definition
void
DumpMultiblockObjects(DBfile *db, PMPIO_baton_t *bat, 
                      char basename[], int numRanks);

// Callback prototypes for PMPIO interface (only useful if we're
// running parallel)
static void *
LULESH_PMPIO_Create(const char *fname,
		     const char *dname,
		     void *udata);
static void *
LULESH_PMPIO_Open(const char *fname,
		   const char *dname,
		   PMPIO_iomode_t ioMode,
		   void *udata);
static void
LULESH_PMPIO_Close(void *file, void *udata);

#else
void
DumpMultiblockObjects(DBfile *db, char basename[], int numRanks);
#endif


/**********************************************************************/
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks) 
{
  char subdirName[32];
  char basename[32];
  DBfile *db;


  sprintf(basename, "lulesh_plot_c%d", domain.cycle());
  sprintf(subdirName, "data_%d", myRank);

#if USE_MPI

  PMPIO_baton_t *bat = PMPIO_Init(numFiles,
				  PMPIO_WRITE,
				  MPI_COMM_WORLD,
				  10101,
				  LULESH_PMPIO_Create,
				  LULESH_PMPIO_Open,
				  LULESH_PMPIO_Close,
				  NULL);

  int myiorank = PMPIO_GroupRank(bat, myRank);

  char fileName[64];
  
  if (myiorank == 0) 
    strcpy(fileName, basename);
  else
    sprintf(fileName, "%s.%03d", basename, myiorank);

  db = (DBfile*)PMPIO_WaitForBaton(bat, fileName, subdirName);

  DumpDomainToVisit(db, domain, myRank);

  // Processor 0 writes out bit of extra data to its file that
  // describes how to stitch all the pieces together
  if (myRank == 0) {
    DumpMultiblockObjects(db, bat, basename, numRanks);
  }

  PMPIO_HandOffBaton(bat, db);

  PMPIO_Finish(bat);
#else

  db = (DBfile*)DBCreate(basename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5X);

  if (db) {
     DBMkDir(db, subdirName);
     DBSetDir(db, subdirName);
     DumpDomainToVisit(db, domain, myRank);
     DumpMultiblockObjects(db, basename, numRanks);
     DBClose(db);
  }
  else {
     printf("Error writing out viz file - rank %d\n", myRank);
  }

#endif
}



/**********************************************************************/

static void 
DumpDomainToVisit(DBfile *db, Domain& domain, int myRank)
{
   int ok = 0;
   
   /* Create an option list that will give some hints to VisIt for
    * printing out the cycle and time in the annotations */
   DBoptlist *optlist;


   /* Write out the mesh connectivity in fully unstructured format */
   int shapetype[1] = {DB_ZONETYPE_HEX};
   int shapesize[1] = {8};
   int shapecnt[1] = {domain.numElem()};
   Kokkos::View<int**, Kokkos::LayoutRight> conn("connectivity", domain.numElem(), 8);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numElem()), KOKKOS_LAMBDA(int ei) {
      for (int n=0; n<8; ++n) {
         conn(ei,n) = domain.m_nodelist(ei,n);
         }
      }
   );

   auto conn_h = Kokkos::create_mirror_view(conn);
   Kokkos::deep_copy(conn_h, conn);
   ok += DBPutZonelist2(db, "connectivity", domain.numElem(), 3,
                        conn_h.data(), domain.numElem()*8,
                        0,0,0, /* Not carrying ghost zones */
                        shapetype, shapesize, shapecnt,
                        1, NULL);
   

   /* Write out the mesh coordinates associated with the mesh */

   Kokkos::View<float*> coords_X("X" , domain.numNode()) ;
   Kokkos::View<float*> coords_Y("Y" , domain.numNode()) ;
   Kokkos::View<float*> coords_Z("Z" , domain.numNode()) ;

   Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numNode()),
     KOKKOS_LAMBDA (int i) {
         coords_X(i) = float(domain.x(i));
         coords_Y(i) = float(domain.y(i));
         coords_Z(i) = float(domain.z(i));
         
     }
   );
   auto X_h = Kokkos::create_mirror_view(coords_X);
   auto Y_h = Kokkos::create_mirror_view(coords_Y);
   auto Z_h = Kokkos::create_mirror_view(coords_Z);
   Kokkos::deep_copy(X_h, coords_X);
   Kokkos::deep_copy(Y_h, coords_Y);   
   Kokkos::deep_copy(Z_h, coords_Z);
   float *coords[3] = {X_h.data(), Y_h.data(), Z_h.data()};
   char *coordnames[3] = {(char*)"X", (char*)"Y", (char*)"Z"};
   
   optlist = DBMakeOptlist(2);
   ok += DBAddOption(optlist, DBOPT_DTIME, &domain.time());
   ok += DBAddOption(optlist, DBOPT_CYCLE, &domain.cycle());

   ok += DBPutUcdmesh(db, "mesh", domain.numNode(), 3,
                      coordnames, coords,
                      domain.numElem(), "connectivity",
                      NULL, DB_FLOAT, optlist);
   ok += DBFreeOptlist(optlist);

   /* Write out the materials */
  /*int *matnums = new int[domain.numReg()];*/
   int dims[1] = {domain.numElem()}; // No mixed elements
   Kokkos::View<int*> matnums("matnums", domain.numReg()) ;
   Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numReg()),
     KOKKOS_LAMBDA (int i) {
         matnums(i) = domain.matElemlist(i) ;
     }
   );
   auto matnums_h = Kokkos::create_mirror_view(matnums);
   Kokkos::deep_copy(matnums_h, matnums);
   
   ok += DBPutMaterial(db, "regions", "mesh", domain.numReg(),
                       matnums_h.data(), domain.regNumList(), dims, 1,
                       NULL, NULL, NULL, NULL, 0, DB_FLOAT, NULL);

   /* Write out pressure, energy, relvol, q */

  // float *e = new float[domain.numElem()] ; 
  Kokkos::View<float*> e("e", domain.numElem()) ;
   Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numElem()),
     KOKKOS_LAMBDA (int i) {
         e(i) = float(domain.e(i)) ;
     }
   );
   auto e_h = Kokkos::create_mirror_view(e);
   Kokkos::deep_copy(e_h, e); 
   ok += DBPutUcdvar1(db, "e", "mesh", e_h.data(),
                      domain.numElem(), NULL, 0, DB_FLOAT, DB_ZONECENT,
                      NULL);
   


   //float *p = new float[domain.numElem()] ; 
   Kokkos::View<float*> p("p", domain.numElem()) ;
      Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numElem()),
     KOKKOS_LAMBDA (int i) {
         p(i) = float(domain.p(i)) ;
     }
   );
   auto p_h = Kokkos::create_mirror_view(p);
   Kokkos::deep_copy(p_h, p);

   ok += DBPutUcdvar1(db, "p", "mesh", p_h.data(),
                      domain.numElem(), NULL, 0, DB_FLOAT, DB_ZONECENT,
                      NULL);


   //float *v = new float[domain.numElem()] ; 
   Kokkos::View<float*> v("v", domain.numElem()) ;
      Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numElem()),
     KOKKOS_LAMBDA (int i) {
         v(i) = float(domain.v(i)) ;
     }
   );
   auto v_h = Kokkos::create_mirror_view(v); 
   Kokkos::deep_copy(v_h, v);
   ok += DBPutUcdvar1(db, "v", "mesh", v_h.data(),
                      domain.numElem(), NULL, 0, DB_FLOAT, DB_ZONECENT,
                      NULL);
   

   //float *q = new float[domain.numElem()] ; 
   Kokkos::View<float*> q("q", domain.numElem()) ;
      Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numElem()),
     KOKKOS_LAMBDA (int i) {
         q(i) = float(domain.q(i)) ;
     }
   );
   auto q_h = Kokkos::create_mirror_view(q); 
   Kokkos::deep_copy(q_h, q);

   ok += DBPutUcdvar1(db, "q", "mesh", q_h.data(),
                      domain.numElem(), NULL, 0, DB_FLOAT, DB_ZONECENT,
                      NULL);
   

   /* Write out nodal speed, velocities */
   /*float *zd    = new float[domain.numNode()];
   float *yd    = new float[domain.numNode()];
   float *xd    = new float[domain.numNode()];
   float *speed = new float[domain.numNode()];*/
   Kokkos::View<float*> zd("zd", domain.numNode()) ;
   Kokkos::View<float*> yd("yd", domain.numNode()) ;
   Kokkos::View<float*> xd("xd", domain.numNode()) ;
   Kokkos::View<float*> speed("speed", domain.numNode()) ;
   Kokkos::parallel_for(
     Kokkos::RangePolicy<Kokkos::Serial>(0, domain.numNode()),
     KOKKOS_LAMBDA (int i) {
         xd(i)    = float(domain.xd(i)) ;
         yd(i)    = float(domain.yd(i)) ;
         zd(i)    = float(domain.zd(i)) ;
         speed(i) = float(sqrt(domain.xd(i)*domain.xd(i) + domain.yd(i)*domain.yd(i) + domain.zd(i)*domain.zd(i))) ;
     }
   );
   auto zd_h = Kokkos::create_mirror_view(zd); 
   auto yd_h = Kokkos::create_mirror_view(yd); 
   auto xd_h = Kokkos::create_mirror_view(xd); 
   auto speed_h = Kokkos::create_mirror_view(speed); 
   Kokkos::deep_copy(zd_h, zd); 
   Kokkos::deep_copy(yd_h, yd); 
   Kokkos::deep_copy(xd_h, xd); 
   Kokkos::deep_copy(speed_h, speed);


   ok += DBPutUcdvar1(db, "speed", "mesh", speed_h.data(),
                      domain.numNode(), NULL, 0, DB_FLOAT, DB_NODECENT,
                      NULL);
   


   ok += DBPutUcdvar1(db, "xd", "mesh", xd_h.data(),
                      domain.numNode(), NULL, 0, DB_FLOAT, DB_NODECENT,
                      NULL);
   

   ok += DBPutUcdvar1(db, "yd", "mesh", yd_h.data(),
                      domain.numNode(), NULL, 0, DB_FLOAT, DB_NODECENT,
                      NULL);


   ok += DBPutUcdvar1(db, "zd", "mesh", zd_h.data(),
                      domain.numNode(), NULL, 0, DB_FLOAT, DB_NODECENT,
                      NULL);
   


   if (ok != 0) {
      printf("Error writing out viz file - rank %d\n", myRank);
   }
}

/**********************************************************************/

#if USE_MPI     
void
   DumpMultiblockObjects(DBfile *db, PMPIO_baton_t *bat, 
                         char basename[], int numRanks)
#else
void
  DumpMultiblockObjects(DBfile *db, char basename[], int numRanks)
#endif
{
   /* MULTIBLOCK objects to tie together multiple files */
  char **multimeshObjs;
  char **multimatObjs;
  char ***multivarObjs;
  int *blockTypes;
  int *varTypes;
  int ok = 0;
  // Make sure this list matches what's written out above
  char vars[][10] = {"p","e","v","q", "speed", "xd", "yd", "zd"};
  int numvars = sizeof(vars)/sizeof(vars[0]);

  // Reset to the root directory of the silo file
  DBSetDir(db, "/");

  // Allocate a bunch of space for building up the string names
  multimeshObjs = new char*[numRanks];
  multimatObjs = new char*[numRanks];
  multivarObjs = new char**[numvars];
  blockTypes = new int[numRanks];
  varTypes = new int[numRanks];

  for(int v=0 ; v<numvars ; ++v) {
     multivarObjs[v] = new char*[numRanks];
  }
  
  for(int i=0 ; i<numRanks ; ++i) {
     multimeshObjs[i] = new char[64];
     multimatObjs[i] = new char[64];
     for(int v=0 ; v<numvars ; ++v) {
        multivarObjs[v][i] = new char[64];
     }
     blockTypes[i] = DB_UCDMESH;
     varTypes[i] = DB_UCDVAR;
  }
      
  // Build up the multiobject names
  for(int i=0 ; i<numRanks ; ++i) {
#if USE_MPI     
    int iorank = PMPIO_GroupRank(bat, i);
#else
    int iorank = 0;
#endif

    //delete multivarObjs[i];
    if (iorank == 0) {
      snprintf(multimeshObjs[i], 64, "/data_%d/mesh", i);
      snprintf(multimatObjs[i], 64, "/data_%d/regions",i);
      for(int v=0 ; v<numvars ; ++v) {
	snprintf(multivarObjs[v][i], 64, "/data_%d/%s", i, vars[v]);
      }
     
    }
    else {
      snprintf(multimeshObjs[i], 64, "%s.%03d:/data_%d/mesh",
               basename, iorank, i);
      snprintf(multimatObjs[i], 64, "%s.%03d:/data_%d/regions", 
	       basename, iorank, i);
      for(int v=0 ; v<numvars ; ++v) {
         snprintf(multivarObjs[v][i], 64, "%s.%03d:/data_%d/%s", 
                  basename, iorank, i, vars[v]);
      }
    }
  }

  // Now write out the objects
  ok += DBPutMultimesh(db, "mesh", numRanks,
		       (char**)multimeshObjs, blockTypes, NULL);
  ok += DBPutMultimat(db, "regions", numRanks,
		      (char**)multimatObjs, NULL);
  for(int v=0 ; v<numvars ; ++v) {
     ok += DBPutMultivar(db, vars[v], numRanks,
                         (char**)multivarObjs[v], varTypes, NULL);
  }

  for(int v=0; v < numvars; ++v) {
    for(int i = 0; i < numRanks; i++) {
      delete multivarObjs[v][i];
    }
    delete multivarObjs[v];
  }

  // Clean up
  for(int i=0 ; i<numRanks ; i++) {
    delete multimeshObjs[i];
    delete multimatObjs[i];
  }
  delete [] multimeshObjs;
  delete [] multimatObjs;
  delete [] multivarObjs;
  delete [] blockTypes;
  delete [] varTypes;

  if (ok != 0) {
    printf("Error writing out multiXXX objs to viz file - rank 0\n");
  }
}

# if USE_MPI

/**********************************************************************/

static void *
LULESH_PMPIO_Create(const char *fname,
		     const char *dname,
		     void *udata)
{
   /* Create the file */
   DBfile* db = DBCreate(fname, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5X);

   /* Put the data in a subdirectory, so VisIt only sees the multimesh
    * objects we write out in the base file */
   if (db) {
     DBMkDir(db, dname);
     DBSetDir(db, dname);
   }
   return (void*)db;
}

   
/**********************************************************************/

static void *
LULESH_PMPIO_Open(const char *fname,
		   const char *dname,
		   PMPIO_iomode_t ioMode,
		   void *udata)
{
   /* Open the file */
  DBfile* db = DBOpen(fname, DB_UNKNOWN, DB_APPEND);

   /* Put the data in a subdirectory, so VisIt only sees the multimesh
    * objects we write out in the base file */
   if (db) {
     DBMkDir(db, dname);
     DBSetDir(db, dname);
   }
   return (void*)db;
}

   
/**********************************************************************/

static void
LULESH_PMPIO_Close(void *file, void *udata)
{
  DBfile *db = (DBfile*)file;
  if (db)
    DBClose(db);
}
# endif

   
#else

void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks)
{
   if (myRank == 0) {
      printf("Must enable -DVIZ_MESH at compile time to call DumpDomain\n");
   }
}

#endif

