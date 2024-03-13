///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018-2023, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include <cufft.h>
#include <stdio.h>
#include <memory>
#include "poissonSolver.h"
#include "placerBase.h"
#include "placerObjects.h"
#include <odb/db.h>
#include "utl/Logger.h"
#include "util.h"
#include <iostream>
#include <cmath>
#include <memory>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
// basic vectors
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
// memory related
#include <thrust/copy.h>
#include <thrust/fill.h>
// algorithm related
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

namespace gpl2 {

using namespace std;
using utl::GPL2;


////////////////////////////////////////////////////////
// PlacerBaseCommon
void PlacerBaseCommon::initCUDAKernel()
{
  // calculate the information on the host side
  hInstDCx_.resize(numPlaceInsts_);
  hInstDCy_.resize(numPlaceInsts_);
  int instIdx = 0;
  for (auto& inst : placeInsts_) {
    hInstDCx_[instIdx] = inst->cx();
    hInstDCy_[instIdx] = inst->cy(); 
    instIdx++;
  }

  // allocate the objects on host side
  dInstDCxPtr_ = setThrustVector<int>(numPlaceInsts_, dInstDCx_);
  dInstDCyPtr_ = setThrustVector<int>(numPlaceInsts_, dInstDCy_);

  // copy from host to device
  thrust::copy(hInstDCx_.begin(), hInstDCx_.end(), dInstDCx_.begin());
  thrust::copy(hInstDCy_.begin(), hInstDCy_.end(), dInstDCy_.begin());

  // allocate memory on device side
  dWLGradXPtr_ = setThrustVector<float>(numPlaceInsts_, dWLGradX_);
  dWLGradYPtr_ = setThrustVector<float>(numPlaceInsts_, dWLGradY_);

  // create the wlGradOp
  //wlGradOp_ = new wirelengthOp(std::shared_ptr<PlacerBaseCommon>(this));
  wlGradOp_ = new WirelengthOp(this);
}

void PlacerBaseCommon::freeCUDAKernel()
{
  dInstDCxPtr_ = nullptr;
  dInstDCyPtr_ = nullptr;

  dWLGradXPtr_ = nullptr;
  dWLGradYPtr_ = nullptr;
}

// Update the database information
void PlacerBaseCommon::updateDB()
{
  if (clusterFlag_ == true) {
    updateDBCluster();
    return;
  }
  
  thrust::copy(dInstDCx_.begin(), dInstDCx_.end(), hInstDCx_.begin());
  thrust::copy(dInstDCy_.begin(), dInstDCy_.end(), hInstDCy_.begin());  
 
  int manufactureGird = db_->getTech()->getManufacturingGrid();

  for (auto inst : placeInsts_) {
    const int instId = inst->instId();
    inst->setCenterLocation(
      static_cast<int>(hInstDCx_[instId]) / manufactureGird * manufactureGird,
      static_cast<int>(hInstDCy_[instId]) / manufactureGird * manufactureGird);
    inst->dbSetLocation();
    inst->dbSetPlaced();
  }
}

void PlacerBaseCommon::plotDBCluster(int iter)
{
  return;
  
  const char* rpt_dir = "./dgl_rpt";
  if (std::filesystem::exists(rpt_dir) == false) {
    std::filesystem::create_directory(rpt_dir);
  }

  std::string file_name = std::string(rpt_dir) + "/" + std::to_string(iter) + ".rpt";
  std::ofstream file;
  file.open(file_name);
  thrust::copy(dInstDCx_.begin(), dInstDCx_.end(), hInstDCx_.begin());
  thrust::copy(dInstDCy_.begin(), dInstDCy_.end(), hInstDCy_.begin());  
  file << "core_lx = " << die_.coreLx() << "  "
       << "core_ly = " << die_.coreLy() << "  "
       << "core_ux = " << die_.coreUx() << "  "
       << "core_uy = " << die_.coreUy() << "  "
       << std::endl;

  for (int i = 0; i < hInstDCx_.size(); i++) {
    file << "index = " << i << "  "
         << "cx = " << hInstDCx_[i] << "  "
         << "cy = " << hInstDCy_[i] << "  "
         << "dx = " << placeInsts_[i]->dx() << "  "
         << "dy = " << placeInsts_[i]->dy() << "  "
         << std::endl;
  }  
  file.close();
}


void PlacerBaseCommon::updateDBCluster()
{
  thrust::copy(dInstDCx_.begin(), dInstDCx_.end(), hInstDCx_.begin());
  thrust::copy(dInstDCy_.begin(), dInstDCy_.end(), hInstDCy_.begin());  

  int manufactureGird = db_->getTech()->getManufacturingGrid();
  odb::dbBlock* block = getBlock();
  // insts fill with real instances
  // update the clusters
  odb::dbSet<odb::dbInst> insts = block->getInsts();
  for (odb::dbInst* inst : insts) {
    auto type = inst->getMaster()->getType();
    if (!type.isCore() && !type.isBlock()) {
      continue;
    }
    const int clusterId = odb::dbIntProperty::find(inst, "cluster_id")->getValue();
    const int cx = hInstDCx_[clusterId];
    const int cy = hInstDCy_[clusterId];
    odb::dbBox* bbox = inst->getBBox();
    int width = bbox->getDX();
    int height = bbox->getDY();
    int lx = cx - width / 2;
    int ly = cy - height / 2;
    if (lx < die().coreLx()) {
      lx = die().coreLx();
    }

    if (ly < die().coreLy()) {
      ly = die().coreLy();
    }
    
    if (lx + width > die().coreUx()) {
      lx = die().coreUx() - width;
    }

    if (ly + height > die().coreUy()) {
      ly = die().coreUy() - height;
    }
    
    lx = lx / manufactureGird * manufactureGird;
    ly = ly / manufactureGird * manufactureGird;
    
    inst->setLocation(lx, ly);
    inst->setPlacementStatus(odb::dbPlacementStatus::PLACED);
  }
}

void PlacerBaseCommon::printInstPinGrad(int instId) const
{
  wlGradOp_->printInstPinGrad(instId);
}

void PlacerBaseCommon::printWireLengthGrad() const {
  wlGradOp_->printNetInfo();    
  for (int i = 0; i < numPlaceInsts_; i++) {
        std::cout << "InstId = " << i << "  "
                  << "wireLengthX = " << dWLGradX_[i] << "  "
                  << "wireLengthY = " << dWLGradY_[i] << "  "
                  << std::endl;
  }
}



////////////////////////////////////////////////////////
// PlacerBase
void PlacerBase::initCUDAKernel()
{
  // calculate the information on the host side
  thrust::host_vector<int> hPlaceInstIds(numPlaceInsts_);
  int instIdx = 0;
  for (auto& inst : placeInsts_) {
    hPlaceInstIds[instIdx] = inst->instId();
    instIdx++;
  }

  // allocate the objects on host side
  dPlaceInstIdsPtr_ = setThrustVector<int>(numPlaceInsts_, dPlaceInstIds_);

  // copy from host to device
  thrust::copy(hPlaceInstIds.begin(), hPlaceInstIds.end(), dPlaceInstIds_.begin());

  thrust::host_vector<int> hInstDDx(numInsts_);
  thrust::host_vector<int> hInstDDy(numInsts_);
  thrust::host_vector<int> hInstDCx(numInsts_);
  thrust::host_vector<int> hInstDCy(numInsts_);
  thrust::host_vector<float> hWireLengthPrecondi(numInsts_);
  thrust::host_vector<float> hDensityPrecondi(numInsts_);

  // calculate the information on the host side
  instIdx = 0;
  for (auto& inst : insts_) {
    hInstDDx[instIdx] = inst->dDx();
    hInstDDy[instIdx] = inst->dDy();
    hInstDCx[instIdx] = inst->cx();
    hInstDCy[instIdx] = inst->cy();
    hWireLengthPrecondi[instIdx] = inst->wireLengthPreconditioner();
    hDensityPrecondi[instIdx] = inst->densityPreconditioner();
    instIdx++;
  } 

  dInstDDxPtr_ = setThrustVector<int>(numInsts_, dInstDDx_);
  dInstDDyPtr_ = setThrustVector<int>(numInsts_, dInstDDy_);
  dInstDCxPtr_ = setThrustVector<int>(numInsts_, dInstDCx_);
  dInstDCyPtr_ = setThrustVector<int>(numInsts_, dInstDCy_);

  dWireLengthPrecondiPtr_ = setThrustVector<float>(numInsts_, dWireLengthPrecondi_);
  dDensityPrecondiPtr_ = setThrustVector<float>(numInsts_, dDensityPrecondi_);
  
  thrust::copy(hInstDDx.begin(), hInstDDx.end(), dInstDDx_.begin());
  thrust::copy(hInstDDy.begin(), hInstDDy.end(), dInstDDy_.begin());
  thrust::copy(hInstDCx.begin(), hInstDCx.end(), dInstDCx_.begin());
  thrust::copy(hInstDCy.begin(), hInstDCy.end(), dInstDCy_.begin());
  thrust::copy(hWireLengthPrecondi.begin(), hWireLengthPrecondi.end(), dWireLengthPrecondi_.begin());
  thrust::copy(hDensityPrecondi.begin(), hDensityPrecondi.end(), dDensityPrecondi_.begin());

  dDensityGradXPtr_ = setThrustVector<float>(
      numInsts_, dDensityGradX_);
  dDensityGradYPtr_ = setThrustVector<float>(
      numInsts_, dDensityGradY_);
    
  dCurSLPCoordiPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dCurSLPCoordi_);
  dCurSLPWireLengthGradXPtr_ = setThrustVector<float>(
      numInsts_, dCurSLPWireLengthGradX_);
  dCurSLPWireLengthGradYPtr_ = setThrustVector<float>(
      numInsts_, dCurSLPWireLengthGradY_);
  dCurSLPDensityGradXPtr_ = setThrustVector<float>( 
      numInsts_, dCurSLPDensityGradX_);
  dCurSLPDensityGradYPtr_ = setThrustVector<float>(
      numInsts_, dCurSLPDensityGradY_);
  dCurSLPSumGradsPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dCurSLPSumGrads_);
    
  dPrevSLPCoordiPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dPrevSLPCoordi_);
  dPrevSLPWireLengthGradXPtr_ = setThrustVector<float>(
      numInsts_, dPrevSLPWireLengthGradX_);
  dPrevSLPWireLengthGradYPtr_ = setThrustVector<float>(
      numInsts_, dPrevSLPWireLengthGradY_);
  dPrevSLPDensityGradXPtr_ = setThrustVector<float>(
      numInsts_, dPrevSLPDensityGradX_);
  dPrevSLPDensityGradYPtr_ = setThrustVector<float>(
      numInsts_, dPrevSLPDensityGradY_);
  dPrevSLPSumGradsPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dPrevSLPSumGrads_);
    
  dNextSLPCoordiPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dNextSLPCoordi_);
  dNextSLPWireLengthGradXPtr_ = setThrustVector<float>(
      numInsts_, dNextSLPWireLengthGradX_);
  dNextSLPWireLengthGradYPtr_ = setThrustVector<float>(
      numInsts_, dNextSLPWireLengthGradY_);
  dNextSLPDensityGradXPtr_ = setThrustVector<float>(
      numInsts_, dNextSLPDensityGradX_);
  dNextSLPDensityGradYPtr_ = setThrustVector<float>(
      numInsts_, dNextSLPDensityGradY_);
  dNextSLPSumGradsPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dNextSLPSumGrads_);

  dCurCoordiPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dCurCoordi_);
  dNextCoordiPtr_ = setThrustVector<FloatPoint>(
      numInsts_, dNextCoordi_);

  dSumGradsXPtr_ = setThrustVector<float>(
      numInsts_, dSumGradsX_);
  dSumGradsYPtr_ = setThrustVector<float>(
      numInsts_, dSumGradsY_);

  densityOp_ = new DensityOp(this);
}


void PlacerBase::freeCUDAKernel()
{
  densityOp_ = nullptr;
  dPlaceInstIdsPtr_ = nullptr;
  
  dDensityGradXPtr_ = nullptr;
  dDensityGradYPtr_ = nullptr;

  dCurSLPCoordiPtr_ = nullptr;
  dCurSLPSumGradsPtr_ = nullptr;

  dPrevSLPCoordiPtr_ = nullptr;
  dPrevSLPSumGradsPtr_ = nullptr;

  dNextSLPCoordiPtr_ = nullptr;
  dNextSLPSumGradsPtr_ = nullptr;

  dCurSLPCoordiPtr_ = nullptr;
  dCurSLPSumGradsPtr_ = nullptr;
}


// Make sure the instances are within the region 
__device__
float getDensityCoordiLayoutInside(
  int instWidth,
  float cx,
  int coreLx,
  int coreUx)
{
  float adjVal = cx;
  if (cx - instWidth / 2 < coreLx) {
    adjVal = coreLx + instWidth / 2;
  }

  if (cx + instWidth / 2 > coreUx) {
    adjVal = coreUx - instWidth / 2;
  }

  return adjVal;
}


__global__
void updateDensityCoordiLayoutInsideKernel(
  const int numInsts,
  const int coreLx,
  const int coreLy,
  const int coreUx,
  const int coreUy,
  const int* instDDx,
  const int* instDDy,
  int* instDCx,
  int* instDCy)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    instDCx[instIdx] = getDensityCoordiLayoutInside(
          instDDx[instIdx], instDCx[instIdx], coreLx, coreUx);
    instDCy[instIdx] = getDensityCoordiLayoutInside(
          instDDy[instIdx], instDCy[instIdx], coreLy, coreUy);
  }
}

__global__
void initDensityCoordiKernel(
  int numInsts,
  const int* instDCx,
  const int* instDCy,
  FloatPoint* dCurCoordiPtr,
  FloatPoint* dCurSLPCoordiPtr,
  FloatPoint* dPrevSLPCoordiPtr)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    const FloatPoint loc(instDCx[instIdx], instDCy[instIdx]);
    dCurCoordiPtr[instIdx] = loc;
    dCurSLPCoordiPtr[instIdx] = loc;
    dPrevSLPCoordiPtr[instIdx] = loc;
  }
}


void PlacerBase::initDensity1()
{
  // update density coordinate for each instance
  int numThreads = 256;
  int numBlocks = (numInsts_ + numThreads - 1) / numThreads;
  updateDensityCoordiLayoutInsideKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    bg_.lx(),
    bg_.ly(),
    bg_.ux(),
    bg_.uy(),
    dInstDDxPtr_,
    dInstDDyPtr_,
    dInstDCxPtr_,
    dInstDCyPtr_
  );

  // initialize the dCurSLPCoordiPtr_, dPrevSLPCoordiPtr_
  // and dCurCoordiPtr_
  initDensityCoordiKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    dInstDCxPtr_,
    dInstDCyPtr_,
    dCurCoordiPtr_,
    dCurSLPCoordiPtr_,
    dPrevSLPCoordiPtr_);
   
      
  // We need to sync up bewteen pb and pbCommon
  updateGCellDensityCenterLocation(dCurSLPCoordiPtr_);
  pbCommon_->updatePinLocation();
  // calculate the previous hpwl
  // update the location of instances within this region
  // while the instances in other regions will not change
  prevHpwl_ = pbCommon_->hpwl();

  std::cout << "prevHpwl_ = " << prevHpwl_ << std::endl;

  // FFT update
  updateDensityForceBin();

  // update parameters
  baseWireLengthCoef_ = npVars_.initWireLengthCoef
                / (static_cast<float>(binSizeX() + binSizeY()) * 0.5);

  sumOverflow_ = static_cast<float>(overflowArea())
                / static_cast<float>(nesterovInstsArea());

  sumOverflowUnscaled_ = static_cast<float>(overflowAreaUnscaled())
                / static_cast<float>(nesterovInstsArea());
  
  std::cout << "[Test a] sumOverflow_ = " << sumOverflow_ << std::endl;
  std::cout << "[Test a] sumOverflowUnscaled_ = " << sumOverflowUnscaled_ << std::endl;
}


// (a)  // (a) define the get distance method
// getDistance is only defined on the host side
struct getTupleDistanceFunctor
{
  __host__ __device__
  float operator()(const thrust::tuple<FloatPoint, FloatPoint>& t) {
    const FloatPoint& a = thrust::get<0>(t);
    const FloatPoint& b = thrust::get<1>(t);
    float dist = 0.0f;
    dist += (a.x - b.x) * (a.x - b.x);
    dist += (a.y - b.y) * (a.y - b.y);
    return dist;
  }
};


__host__
float getDistance(const FloatPoint* a, 
                  const FloatPoint* b, 
                  const int numInsts)
{
  if (numInsts <= 0) {
    return 0.0;
  }

  thrust::device_ptr<FloatPoint> aBegin(const_cast<FloatPoint*>(a));
  thrust::device_ptr<FloatPoint> aEnd = aBegin + numInsts;

  thrust::device_ptr<FloatPoint> bBegin(const_cast<FloatPoint*>(b));
  thrust::device_ptr<FloatPoint> bEnd = bBegin + numInsts;

  float sumDistance = thrust::transform_reduce(
    thrust::make_zip_iterator(thrust::make_tuple(aBegin, bBegin)),
    thrust::make_zip_iterator(thrust::make_tuple(aEnd, bEnd)),
    getTupleDistanceFunctor(),
    0.0f,
    thrust::plus<float>()
  );

  return std::sqrt(sumDistance / (2.0 * numInsts));
}


template<typename T>
struct myAbs
{
  __host__ __device__
  double operator() (const T& x) const
  {
    if(x >= 0)
      return x;
    else
      return x * -1;
  }
};


__host__ 
float getAbsGradSum(const float* a, const int numInsts)
{
  thrust::device_ptr<float> aBegin(const_cast<float*>(a));
  thrust::device_ptr<float> aEnd = aBegin + numInsts;
  double sumAbs = thrust::transform_reduce(
    aBegin,
    aEnd, 
    myAbs<float>(), 
    0.0, 
    thrust::plus<double>()
  ); 
  return sumAbs;
}


float PlacerBase::getStepLength(
  const FloatPoint* prevSLPCoordi,
  const FloatPoint* prevSLPSumGrads,
  const FloatPoint* curSLPCoordi,
  const FloatPoint* curSLPSumGrads) const
{
  float coordiDistance = getDistance(prevSLPCoordi, curSLPCoordi, numInsts_);
  float gradDistance = getDistance(prevSLPSumGrads, curSLPSumGrads, numInsts_);
  return coordiDistance / gradDistance;
}

// Function: initDensity2
float PlacerBase::initDensity2()
{
  // the wirelength force on each instance is zero
  if (wireLengthGradSum_ == 0) {
    densityPenalty_ = npVars_.initDensityPenalty;
    updatePrevGradient();
  }

  if (wireLengthGradSum_ != 0) {
    densityPenalty_
        = (wireLengthGradSum_ / densityGradSum_) * npVars_.initDensityPenalty;
  }

  sumOverflow_ = static_cast<float>(overflowArea())
                 / static_cast<float>(nesterovInstsArea());

  sumOverflowUnscaled_ = static_cast<float>(overflowAreaUnscaled())
                         / static_cast<float>(nesterovInstsArea());
  
  stepLength_ = getStepLength(
      dPrevSLPCoordiPtr_, 
      dPrevSLPSumGradsPtr_, 
      dCurSLPCoordiPtr_, 
      dCurSLPSumGradsPtr_);


  /*
  // for debug
  thrust::device_vector<FloatPoint> dPrevSLPSumGrads(numInsts_);
  thrust::device_vector<FloatPoint> dCurSLPSumGrads(numInsts_);
  thrust::copy(dPrevSLPSumGradsPtr_, dPrevSLPSumGradsPtr_ + numInsts_, dPrevSLPSumGrads.begin());
  thrust::copy(dCurSLPSumGradsPtr_, dCurSLPSumGradsPtr_ + numInsts_, dCurSLPSumGrads.begin());
  thrust::host_vector<FloatPoint> hPrevSLPSumGrads = dPrevSLPSumGrads;
  thrust::host_vector<FloatPoint> hCurSLPSumGrads = dCurSLPSumGrads;
  for (int i = 0; i < numInsts_; i++) {
    std::cout << "idx = " << i << "  "
              << "prevSLPSumGradsX = " << hPrevSLPSumGrads[i].x << "  "
              << "prevSLPSumGradsY = " << hPrevSLPSumGrads[i].y << "  "
              << "curSLPSumGradsX = " << hCurSLPSumGrads[i].x << "  "
              << "curSLPSumGradsY = " << hCurSLPSumGrads[i].y << "  "
              << std::endl;
  }
  */

  return stepLength_;
}


__global__
void sumGradientKernel(
  const int numInsts,
  const float densityPenalty,
  const float minPrecondi,
  const float* wireLengthPrecondi,
  const float* densityPrecondi,
  const float* wireLengthGradientsX,
  const float* wireLengthGradientsY,
  const float* densityGradientsX,
  const float* densityGradientsY,
  FloatPoint* sumGrads)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    sumGrads[instIdx].x = wireLengthGradientsX[instIdx] + densityPenalty * densityGradientsX[instIdx];
    sumGrads[instIdx].y = wireLengthGradientsY[instIdx] + densityPenalty * densityGradientsY[instIdx];
    FloatPoint sumPrecondi(
      wireLengthPrecondi[instIdx] + densityPenalty * densityPrecondi[instIdx],
      wireLengthPrecondi[instIdx] + densityPenalty * densityPrecondi[instIdx]
    );

    if (sumPrecondi.x < minPrecondi) {
      sumPrecondi.x = minPrecondi;
    }

    if (sumPrecondi.y < minPrecondi) {
      sumPrecondi.y = minPrecondi;
    }

    sumGrads[instIdx].x /= sumPrecondi.x;
    sumGrads[instIdx].y /= sumPrecondi.y;
  }
}

void PlacerBase::updatePrevGradient()
{
  updateGradients(dPrevSLPWireLengthGradXPtr_,
                  dPrevSLPWireLengthGradYPtr_,
                  dPrevSLPDensityGradXPtr_,
                  dPrevSLPDensityGradYPtr_,
                  dPrevSLPSumGradsPtr_);
}

void PlacerBase::updateCurGradient()
{
  updateGradients(dCurSLPWireLengthGradXPtr_,
                  dCurSLPWireLengthGradYPtr_,
                  dCurSLPDensityGradXPtr_,
                  dCurSLPDensityGradYPtr_,
                  dCurSLPSumGradsPtr_);

  /*
  // For debug
  std::cout << std::endl;
  std::cout << "updateCurGradient" << std::endl;
  
  thrust::device_vector<float> dCurSLPWireLengthGradX(numInsts_);
  thrust::device_vector<float> dCurSLPWireLengthGradY(numInsts_);
  thrust::device_vector<float> dCurSLPDensityGradX(numInsts_);
  thrust::device_vector<float> dCurSLPDensityGradY(numInsts_);

  thrust::copy(dCurSLPWireLengthGradXPtr_, dCurSLPWireLengthGradXPtr_ + numInsts_, dCurSLPWireLengthGradX.begin());
  thrust::copy(dCurSLPWireLengthGradYPtr_, dCurSLPWireLengthGradYPtr_ + numInsts_, dCurSLPWireLengthGradY.begin());
  thrust::copy(dCurSLPDensityGradXPtr_, dCurSLPDensityGradXPtr_ + numInsts_, dCurSLPDensityGradX.begin());
  thrust::copy(dCurSLPDensityGradYPtr_, dCurSLPDensityGradYPtr_ + numInsts_, dCurSLPDensityGradY.begin());

  thrust::device_vector<FloatPoint> dCurSLPSumGrads(numInsts_);
  thrust::copy(dCurSLPSumGradsPtr_, dCurSLPSumGradsPtr_ + numInsts_, dCurSLPSumGrads.begin());

  thrust::host_vector<float> hCurSLPWireLengthGradX = dCurSLPWireLengthGradX;
  thrust::host_vector<float> hCurSLPWireLengthGradY = dCurSLPWireLengthGradY;
  thrust::host_vector<float> hCurSLPDensityGradX = dCurSLPDensityGradX;
  thrust::host_vector<float> hCurSLPDensityGradY = dCurSLPDensityGradY;
  thrust::host_vector<FloatPoint> hCurSLPSumGrads = dCurSLPSumGrads;

  for (int i = 0; i < numInsts_; i++) {
    std::cout << "a.x = " << hCurSLPWireLengthGradX[i] << "  "
              << "b.x = " << hCurSLPDensityGradX[i] << "  "
              << "c.x = " << hCurSLPSumGrads[i].x << "  "
              << std::endl;
    
    std::cout << "a.y = " << hCurSLPWireLengthGradY[i] << "  "
              << "b.y = " << hCurSLPDensityGradY[i] << "  "
              << "c.y = " << hCurSLPSumGrads[i].y << "  "
              << std::endl;
  }
  */
}

void PlacerBase::updateNextGradient()
{
  updateGradients(dNextSLPWireLengthGradXPtr_,
                  dNextSLPWireLengthGradYPtr_,
                  dNextSLPDensityGradXPtr_,
                  dNextSLPDensityGradYPtr_,
                  dNextSLPSumGradsPtr_);
}

void PlacerBase::updateGradients(
  float* wireLengthGradientsX,
  float* wireLengthGradientsY,
  float* densityGradientsX,
  float* densityGradientsY,
  FloatPoint* sumGrads)
{
  if (isConverged_) {
    return;
  }

  auto startTimestamp = std::chrono::high_resolution_clock::now();

  wireLengthGradSum_ = 0;
  densityGradSum_ = 0;

  // get the forces on each instance
  getWireLengthGradientWA(wireLengthGradientsX, wireLengthGradientsY);
  getDensityGradient(densityGradientsX, densityGradientsY);
  
  auto endTimestamp1 = std::chrono::high_resolution_clock::now();
  double endTime1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTimestamp1- startTimestamp).count();

  updateGradientRuntime1_ += endTime1 * 1e-9;   


  if (false) {
    // for debug
    // What will happen if we copy the data into host side
    thrust::device_vector<float> dDensityGradientsX(densityGradientsX, densityGradientsX + numInsts_);
    thrust::device_vector<float> dDensityGradientsY(densityGradientsY, densityGradientsY + numInsts_);

    thrust::host_vector<float> hDensityGradientsX = dDensityGradientsX;
    thrust::host_vector<float> hDensityGradientsY = dDensityGradientsY;

  }

  wireLengthGradSum_ += getAbsGradSum(wireLengthGradientsX, numInsts_);
  wireLengthGradSum_ += getAbsGradSum(wireLengthGradientsY, numInsts_); 
  densityGradSum_ += getAbsGradSum(densityGradientsX, numInsts_);
  densityGradSum_ += getAbsGradSum(densityGradientsY, numInsts_);

  //std::cout << "wireLengthGradSum_ = " << wireLengthGradSum_ << std::endl;
  //std::cout << "densityGradSum_ = " << densityGradSum_ << std::endl;

  if (debugFlag_ == true) {
    // for debug
    std::cout << "numInsts_ = " << numInsts_ << std::endl;
 

  // for debug
  // What will happen if we copy the data into host side
  thrust::device_vector<float> dDensityGradientsX(densityGradientsX, densityGradientsX + numInsts_);
  thrust::device_vector<float> dDensityGradientsY(densityGradientsY, densityGradientsY + numInsts_);

  thrust::host_vector<float> hDensityGradientsX = dDensityGradientsX;
  thrust::host_vector<float> hDensityGradientsY = dDensityGradientsY;

  thrust::device_vector<float> dWireLengthGradientsX(wireLengthGradientsX, wireLengthGradientsX + numInsts_);
  thrust::device_vector<float> dWireLengthGradientsY(wireLengthGradientsY, wireLengthGradientsY + numInsts_);

  thrust::host_vector<float> hWireLengthGradientsX = dWireLengthGradientsX;
  thrust::host_vector<float> hWireLengthGradientsY = dWireLengthGradientsY;

  densityGradSum_ = 0.0;
  wireLengthGradSum_ = 0.0;
  for (int i = 0; i < numInsts_; i++) {
    densityGradSum_ += fabs(hDensityGradientsX[i]);
    densityGradSum_ += fabs(hDensityGradientsY[i]);
    wireLengthGradSum_ += fabs(hWireLengthGradientsX[i]);
    wireLengthGradSum_ += fabs(hWireLengthGradientsY[i]);

    if (isnan(wireLengthGradSum_) == true) {
      std::cout << "wireLengthGradSum_ is nan" << std::endl;
      std::cout << "instance id = " << i << "  "
                << "hWireLengthGradientsX[i] = " << hWireLengthGradientsX[i] << "  "
                << "hWireLengthGradientsY[i] = " << hWireLengthGradientsY[i] << "  "
                << std::endl;
      pbCommon_->printInstPinGrad(i);
      break;
    }
  }

  }

  auto endTimestamp2 = std::chrono::high_resolution_clock::now();
  double endTime2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTimestamp2- startTimestamp).count();

  updateGradientRuntime2_ += endTime2 * 1e-9;   

  auto endTimestamp3 = std::chrono::high_resolution_clock::now();
  double endTime3 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTimestamp3- startTimestamp).count();
  updateGradientRuntime3_ += endTime3 * 1e-9;   


  auto endTimestamp4 = std::chrono::high_resolution_clock::now();
  double endTime4 = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTimestamp4- startTimestamp).count();
  updateGradientRuntime4_ += endTime4 * 1e-9;   

  int numThreads = 256;
  int numBlocks = (numInsts_ + numThreads - 1) / numThreads;
  sumGradientKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    densityPenalty_,
    npVars_.minPreconditioner,
    dWireLengthPrecondiPtr_,
    dDensityPrecondiPtr_,
    wireLengthGradientsX,
    wireLengthGradientsY,
    densityGradientsX,
    densityGradientsY,
    sumGrads
  );

  /*
  // for debug
  std::cout << std::endl;
  std::cout << "updateGradients" << std::endl;
  thrust::device_vector<float> wireLengthPrecondi(numInsts_);
  thrust::device_vector<float> densityPrecondi(numInsts_);

  thrust::copy(dWireLengthPrecondiPtr_, dWireLengthPrecondiPtr_ + numInsts_, wireLengthPrecondi.begin());
  thrust::copy(dDensityPrecondiPtr_, dDensityPrecondiPtr_ + numInsts_, densityPrecondi.begin());

  thrust::host_vector<float> hWireLengthPrecondi = wireLengthPrecondi;
  thrust::host_vector<float> hDensityPrecondi = densityPrecondi;

  for (int i = 0; i < numInsts_; i++) {
    std::cout << "GCell i = " << i << "  "
              << "densityPenalty_ = " << densityPenalty_ << "  "
              << "wireLengthPrecondi = " << hWireLengthPrecondi[i] << "  "
              << "densityPrecondi = " << hDensityPrecondi[i] << "  "
              << std::endl;
  }
  */

  auto endTimestamp = std::chrono::high_resolution_clock::now();
  double endTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTimestamp- startTimestamp).count();
  updateGradientRuntime_ += endTime * 1e-9;   
}


// sync up the instances location based on the corrodinates
__global__
void updateGCellDensityCenterLocationKernel(
  const int numInsts,
  const FloatPoint* coordis,
  int* instDCx,
  int* instDCy)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    instDCx[instIdx] = coordis[instIdx].x;
    instDCy[instIdx] = coordis[instIdx].y;
  }
}


// sync up the instances between pbCommon and current pb
__global__
void syncPlaceInstsCommonKernel(
  const int numPlaceInsts,
  const int* placeInstIds,
  const int* placeInstDCx,
  const int* placeInstDCy,
  int* instDCxCommon,
  int* instDCyCommon)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numPlaceInsts) {
    int instId = placeInstIds[instIdx];
    instDCxCommon[instId] = placeInstDCx[instIdx];
    instDCyCommon[instId] = placeInstDCy[instIdx];
  }
}


void PlacerBase::updateGCellDensityCenterLocation(const FloatPoint* coordis)
{
  const int numThreads = 256;
  const int numBlocks = (numInsts_ + numThreads - 1) / numThreads;
  const int numPlaceInstBlocks = (numPlaceInsts_ + numThreads - 1) / numThreads;

  updateGCellDensityCenterLocationKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    coordis,
    dInstDCxPtr_,
    dInstDCyPtr_
  );

  syncPlaceInstsCommonKernel<<<numPlaceInstBlocks, numThreads>>>(
    numPlaceInsts_,
    dPlaceInstIdsPtr_,
    dInstDCxPtr_,
    dInstDCyPtr_,
    pbCommon_->dInstDCxPtr(),
    pbCommon_->dInstDCyPtr());

  densityOp_->updateGCellLocation(dInstDCxPtr_, dInstDCyPtr_);
}


__global__
void getWireLengthGradientWAKernel(
  const int numPlaceInsts,
  const int* dPlaceInstIdsPtr,
  const float* dWLGradXCommonPtr,
  const float* dWLGradYCommonPtr,
  float* dWireLengthGradXPtr,
  float* dWireLengthGradYPtr)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numPlaceInsts) {
    int instId = dPlaceInstIdsPtr[instIdx];
    dWireLengthGradXPtr[instIdx] = dWLGradXCommonPtr[instId];
    dWireLengthGradYPtr[instIdx] = dWLGradYCommonPtr[instId];
  }
}





void PlacerBase::getWireLengthGradientWA(
  float* wireLengthGradientsX,
  float* wireLengthGradientsY)
{
  int numThreads = 256;
  int numBlocks = (numPlaceInsts_ + numThreads - 1) / numThreads;

  /*
  // for debug
  thrust::device_vector<float> dWLGradXCommon(numInsts_);
  thrust::device_vector<float> dWLGradYCommon(numInsts_);

  thrust::copy(pbCommon_->dWLGradXPtr(), pbCommon_->dWLGradXPtr() + numPlaceInsts_, dWLGradXCommon.begin());
  thrust::copy(pbCommon_->dWLGradYPtr(), pbCommon_->dWLGradYPtr() + numPlaceInsts_, dWLGradYCommon.begin());

  thrust::host_vector<float> hWLGradXCommon = dWLGradXCommon;
  thrust::host_vector<float> hWLGradYCommon = dWLGradYCommon;

  for (int i = 0; i < numPlaceInsts_; i++) {
    std::cout << "pbCommon  instId = " << i << " "
              << "wirelengthGradX = " << hWLGradXCommon[i] << "  "
              << "wirelengthGradY = " << hWLGradYCommon[i] << "  "
              << "placeInstId = " << dPlaceInstIds_[i] << "  "
              << std::endl;
  }

  std::cout << "numPlaceInsts_ " << numPlaceInsts_ << std::endl;
  */

  getWireLengthGradientWAKernel<<<numBlocks, numThreads>>>(
    numPlaceInsts_,
    dPlaceInstIdsPtr_,
    pbCommon_->dWLGradXPtr(),
    pbCommon_->dWLGradYPtr(),
    wireLengthGradientsX,
    wireLengthGradientsY);

  //wireLengthGradSum_ = getGradSum(wireLengthGradientsX, wireLengthGradientsY, numInsts_);
}

void PlacerBase::getDensityGradient(float* densityGradientsX,
                                    float* densityGradientsY)
{
  densityOp_->getDensityGradient(densityGradientsX, densityGradientsY);
  //densityGradSum_ = getGradSum(densityGradientsX, densityGradientsY, numInsts_);
}

// calculate the next state based on current state
__global__
void nesterovUpdateCooridnatesKernel(
  const int numInsts,
  const int coreLx,
  const int coreLy,
  const int coreUx,
  const int coreUy,
  const float stepLength,
  const float coeff,
  const int* instDDx,
  const int* instDDy,
  const FloatPoint* curCoordiPtr,
  const FloatPoint* curSLPCoordiPtr,
  const FloatPoint* curSLPSumGradsPtr,
  FloatPoint* nextCoordiPtr,
  FloatPoint* nextSLPCoordiPtr)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    FloatPoint nextCoordi(
      curSLPCoordiPtr[instIdx].x + stepLength * curSLPSumGradsPtr[instIdx].x,
      curSLPCoordiPtr[instIdx].y + stepLength * curSLPSumGradsPtr[instIdx].y
    );


    FloatPoint nextSLPCoordi(
      nextCoordi.x + coeff * (nextCoordi.x - curCoordiPtr[instIdx].x),
      nextCoordi.y + coeff * (nextCoordi.y - curCoordiPtr[instIdx].y)
    );
      

    // check the boundary
    nextCoordiPtr[instIdx] = FloatPoint(
      getDensityCoordiLayoutInside(instDDx[instIdx], nextCoordi.x, coreLx, coreUx),
      getDensityCoordiLayoutInside(instDDy[instIdx], nextCoordi.y, coreLy, coreUy)
    );
    
    nextSLPCoordiPtr[instIdx] = FloatPoint(
      getDensityCoordiLayoutInside(instDDx[instIdx], nextSLPCoordi.x, coreLx, coreUx),
      getDensityCoordiLayoutInside(instDDy[instIdx], nextSLPCoordi.y, coreLy, coreUy)
    );
  }
}


void PlacerBase::nesterovUpdateCoordinates(float coeff)
{
  if (isConverged_) {
    return;
  }

  int numThreads = 256;
  int numBlocks = (numInsts_ + numThreads - 1) / numThreads;

  nesterovUpdateCooridnatesKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    bg_.lx(),
    bg_.ly(),
    bg_.ux(),
    bg_.uy(),
    stepLength_,
    coeff,
    dInstDDxPtr_,
    dInstDDyPtr_,
    dCurCoordiPtr_,
    dCurSLPCoordiPtr_,
    dCurSLPSumGradsPtr_,
    dNextCoordiPtr_,
    dNextSLPCoordiPtr_
  );

  // update density
  updateGCellDensityCenterLocation(dNextSLPCoordiPtr_);
  updateDensityForceBin();
}


// exchange the states:  prev -> current, current -> next
// update the parameters
void PlacerBase::updateNextIter(int iter)
{
  if (isConverged_) {
    return;
  }

  // Previous <= Current
  std::swap(dCurSLPCoordiPtr_, dPrevSLPCoordiPtr_);
  std::swap(dCurSLPSumGradsPtr_, dPrevSLPSumGradsPtr_);
  std::swap(dCurSLPWireLengthGradXPtr_, dPrevSLPWireLengthGradXPtr_);
  std::swap(dCurSLPWireLengthGradYPtr_, dPrevSLPWireLengthGradYPtr_);
  std::swap(dCurSLPDensityGradXPtr_, dPrevSLPDensityGradXPtr_);
  std::swap(dCurSLPDensityGradYPtr_, dPrevSLPDensityGradYPtr_);

  // Current <= Next
  std::swap(dCurSLPCoordiPtr_, dNextSLPCoordiPtr_);
  std::swap(dCurSLPSumGradsPtr_, dNextSLPSumGradsPtr_);
  std::swap(dCurSLPWireLengthGradXPtr_, dNextSLPWireLengthGradXPtr_);
  std::swap(dCurSLPWireLengthGradYPtr_, dNextSLPWireLengthGradYPtr_);
  std::swap(dCurSLPDensityGradXPtr_, dNextSLPDensityGradXPtr_);
  std::swap(dCurSLPDensityGradYPtr_, dNextSLPDensityGradYPtr_);
  
  std::swap(dCurCoordiPtr_, dNextCoordiPtr_);
  
  // In a macro dominated design like mock-array-big you may be placing
  // very few std cells in a sea of fixed macros. The overflow denominator
  // may be quite small and prevent convergence. This is mostly due to 
  // our limited ability to move instances off macros cleanly.
  // As that improves this should no longer be needed.
  const float fractionOfMaxIters
      = static_cast<float>(iter) / npVars_.maxNesterovIter;  
  const float overflowDenominator
      = std::max(static_cast<float>(nesterovInstsArea()),
                 fractionOfMaxIters * nonPlaceInstsArea() * 0.05f); 
      
  sumOverflow_ = overflowArea() / overflowDenominator;
  sumOverflowUnscaled_ = overflowAreaUnscaled() / overflowDenominator;

  int64_t hpwl = pbCommon_->hpwl();
  float phiCoef = getPhiCoef(static_cast<float>(hpwl - prevHpwl_)
                             / npVars_.referenceHpwl);

  prevHpwl_ = hpwl;
  //densityPenalty_ *= phiCoef * 1.01;
  densityPenalty_ *= phiCoef * 0.99;

  if (iter == 0 || (iter + 1) % 10 == 0) {
    std::string msg = "[NesterovSolve] Iter: "+ std::to_string(iter + 1) + " ";
    msg += "overflow: " + std::to_string(sumOverflowUnscaled_) + " ";
    msg += "HPWL: " + std::to_string(prevHpwl_) + " ";
    std::cout << msg << " ";
    std::cout << "densityPenalty: " << densityPenalty_ << std::endl;
    //msg += "densityPenalty: " + std::to_string(double(densityPenalty_));
    //log_->report(msg);
  }

  if (iter > 50 && minSumOverflow_ > sumOverflowUnscaled_) {
    minSumOverflow_ = sumOverflowUnscaled_;
    hpwlWithMinSumOverflow_ = prevHpwl_;
  }
}


__global__
void updateInitialPrevSLPCoordiKernel(
  const int numInsts,
  const int coreLx,
  const int coreLy,
  const int coreUx,
  const int coreUy,
  const int* instDDx,
  const int* instDDy,
  const float initialPrevCoordiUpdateCoef,
  const FloatPoint* dCurSLPCoordiPtr,
  const FloatPoint* dCurSLPSumGradsPtr,
  FloatPoint* dPrevSLPCoordiPtr)
{
  int instIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (instIdx < numInsts) {
    const float preCoordiX = dCurSLPCoordiPtr[instIdx].x
                           - initialPrevCoordiUpdateCoef * dCurSLPSumGradsPtr[instIdx].x;
    const float preCoordiY = dCurSLPCoordiPtr[instIdx].y
                           - initialPrevCoordiUpdateCoef * dCurSLPSumGradsPtr[instIdx].y;
    const FloatPoint newCoordi(
      getDensityCoordiLayoutInside(instDDx[instIdx], preCoordiX, coreLx, coreUx),
      getDensityCoordiLayoutInside(instDDy[instIdx], preCoordiY, coreLy, coreUy)
    );
    dPrevSLPCoordiPtr[instIdx] = newCoordi;
  }
}


void PlacerBase::updateInitialPrevSLPCoordi()
{
  const int numThreads = 256;
  const int numBlocks = (numInsts_ + numThreads - 1) / numThreads;
  updateInitialPrevSLPCoordiKernel<<<numBlocks, numThreads>>>(
    numInsts_,
    bg_.lx(),
    bg_.ly(),
    bg_.ux(),
    bg_.uy(),
    dInstDDxPtr_,
    dInstDDyPtr_,
    npVars_.initialPrevCoordiUpdateCoef,
    dCurSLPCoordiPtr_,
    dCurSLPSumGradsPtr_,
    dPrevSLPCoordiPtr_);

  /*
  // for debug
  thrust::device_vector<FloatPoint> dCurSLPCoordi(numInsts_);
  thrust::copy(dCurSLPCoordiPtr_, dCurSLPCoordiPtr_ + numInsts_, dCurSLPCoordi.begin());

  thrust::device_vector<FloatPoint> dCurSLPSumGrads(numInsts_);
  thrust::copy(dCurSLPSumGradsPtr_, dCurSLPSumGradsPtr_ + numInsts_, dCurSLPSumGrads.begin());

  thrust::device_vector<FloatPoint> dPrevSLPCoordi(numInsts_);
  thrust::copy(dPrevSLPCoordiPtr_, dPrevSLPCoordiPtr_ + numInsts_, dPrevSLPCoordi.begin());

  thrust::host_vector<FloatPoint> hCurSLPCoordi = dCurSLPCoordi;
  thrust::host_vector<FloatPoint> hCurSLPSumGrads = dCurSLPSumGrads;
  thrust::host_vector<FloatPoint> hPrevSLPCoordi = dPrevSLPCoordi;

  std::cout << std::endl;
  std::cout << "updateInitialPrevSLPCoordi" << std::endl;
  for (int i = 0; i < numInsts_; i++) {
    std::cout << "a.x = " << hCurSLPCoordi[i].x << "  "
              << "b.x = " << hCurSLPSumGrads[i].x << "  "
              << "c.x = " << hPrevSLPCoordi[i].x << "  "
              << std::endl;
    std::cout << "a.y = " << hCurSLPCoordi[i].y << "  "
              << "b.y = " << hCurSLPSumGrads[i].y << "  "
              << "c.y = " << hPrevSLPCoordi[i].y << "  "
              << std::endl;
  }
  */
}

}
