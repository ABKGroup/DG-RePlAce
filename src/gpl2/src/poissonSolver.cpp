#include <stdio.h>

#include <cmath>
#include <memory>

//#include "PlacerCore.h"
#include "poissonSolver.h"

namespace gpl2 {
PoissonSolver::PoissonSolver()
    : binCntX_(0),
      binCntY_(0),
      binSizeX_(0),
      binSizeY_(0),

      d_expkN_(nullptr),
      d_expkM_(nullptr),

      d_expkNForInverse_(nullptr),
      d_expkMForInverse_(nullptr),

      d_expkMN1_(nullptr),
      d_expkMN2_(nullptr),

      d_binDensity_(nullptr),
      d_auv_(nullptr),
      d_potential_(nullptr),

      d_efX_(nullptr),
      d_efY_(nullptr),

      d_workSpaceReal1_(nullptr),
      d_workSpaceReal2_(nullptr),
      d_workSpaceReal3_(nullptr),
      d_workSpaceComplex_(nullptr),

      d_inputForX_(nullptr),
      d_inputForY_(nullptr)
{
}

PoissonSolver::PoissonSolver(int binCntX, 
                             int binCntY,
                             int binSizeX,
                             int binSizeY) : PoissonSolver()
{
  binCntX_ = binCntX;
  binCntY_ = binCntY;
  binSizeX_ = binSizeX;
  binSizeY_ = binSizeY;

  init();
}

PoissonSolver::~PoissonSolver()
{
  freeDeviceMemory();
}

void PoissonSolver::init()
{
  printf("[PoissonSolver] Start PoissonSolver Initialization!\n");

  setupForCUDAKernel();

  printf("[PoissonSolver] PoissonSolver is initialized!\n");
}

};  // namespace gpl2
