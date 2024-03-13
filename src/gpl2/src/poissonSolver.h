#pragma once

#include <cufft.h>
#include <stdio.h>

#include <memory>
#include "fft.h"

#define FFT_PI 3.141592653589793238462L

namespace gpl2 {

class PoissonSolver
{
 public:
  PoissonSolver();
  PoissonSolver(int binCntX, int binCntY, int binSizeX, int binSizeY);
  ~PoissonSolver();

  // Compute Potential and Electric Force
  // row-major order
  void solvePoisson(const float* binDensity,
                    float* potential,
                    float* electroForceX,
                    float* electroForceY);

  // Compute Potential Only (not Electric Force)
  // row-major order
  void solvePoissonPotential(const float* binDensity, float* potential);

 private:
  int binCntX_;
  int binCntY_;
  int binSizeX_;
  int binSizeY_;

  void init();
  void setupForCUDAKernel();
  void freeDeviceMemory();

  cufftHandle plan_;
  cufftHandle planInverse_;

  cufftComplex* d_expkN_;
  cufftComplex* d_expkM_;

  cufftComplex* d_expkNForInverse_;
  cufftComplex* d_expkMForInverse_;

  cufftComplex* d_expkMN1_;
  cufftComplex* d_expkMN2_;

  cufftReal* d_binDensity_;
  cufftReal* d_auv_;
  cufftReal* d_potential_;

  cufftReal* d_efX_;
  cufftReal* d_efY_;

  cufftReal* d_workSpaceReal1_;
  cufftReal* d_workSpaceReal2_;
  cufftReal* d_workSpaceReal3_;

  cufftComplex* d_workSpaceComplex_;

  cufftReal* d_inputForX_;
  cufftReal* d_inputForY_;
};

};  // namespace gpl2
