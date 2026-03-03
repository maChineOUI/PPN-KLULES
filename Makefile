# =========================================================
# LULESH 2.0 — Clean Serial Build (macOS / Apple Silicon)
# No MPI
# No OpenMP
# No Silo
# =========================================================

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

# Compiler (macOS default)
CXX = clang++

# Disable MPI
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -I. -DUSE_MPI=0
LDFLAGS  = -O3

SOURCES = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc \
	lulesh-geometry.cc \
	lulesh-stress.cc \
	lulesh-nodal.cc \
	lulesh-kinematics.cc \
	lulesh-viscosity.cc \
	lulesh-eos.cc \
	lulesh-timestep.cc \
	lulesh-integration.cc

OBJECTS = $(SOURCES:.cc=.o)

# Compile rule
.cc.o:
	@echo "Building $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link
$(LULESH_EXEC): $(OBJECTS)
	@echo "Linking"
	$(CXX) $(OBJECTS) $(LDFLAGS) -lm -o $@

all: $(LULESH_EXEC)

clean:
	rm -f *.o *~ $(LULESH_EXEC)
	rm -rf *.dSYM
