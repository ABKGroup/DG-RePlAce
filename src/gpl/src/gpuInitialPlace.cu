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

// In our implementation, we use the 
// Biconjugate Gradient Stabilized (BiCGstab) from the cusp
// library to solve the linear equation.
// Check here from example : 
// https://cusplibrary.github.io/group__krylov__methods.html#ga23cfa8325966505d6580151f91525887

#include "gpuInitialPlace.h"
#include <utility>
#include "gpuPlacerBase.h"
#include <cusparse.h>
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
#include <cusp/version.h>
#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/hyb_matrix.h>
#include <cusp/print.h>
#include <cusp/copy.h>
#include <cusp/precond/diagonal.h>
#include <cusp/blas/blas.h>
#include <cusp/coo_matrix.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>


namespace gpl2 {
using namespace std;

InitialPlaceVars::InitialPlaceVars()
{
  reset();
}

void InitialPlaceVars::reset()
{
  maxIter = 20;
  minDiffLength = 1500;
  maxSolverIter = 100;
  maxFanout = 200;
  netWeightScale = 800.0;
}


GpuInitialPlace::GpuInitialPlace() : pbc_(nullptr), log_(nullptr)
{
}

GpuInitialPlace::GpuInitialPlace(InitialPlaceVars ipVars,
                                 std::shared_ptr<GpuPlacerBaseCommon> pbc,
                                 std::vector<std::shared_ptr<GpuPlacerBase> >& pbVec,
                                 utl::Logger* log)
    : ipVars_(ipVars), pbc_(std::move(pbc)), pbVec_(pbVec), log_(log)
{
}

GpuInitialPlace::~GpuInitialPlace()
{
  reset();
}


void GpuInitialPlace::doBicgstabPlace()
{
  std::cout << "GpuInitialPlace :: doBicgstabPlace" << std::endl;
  placeInstsCenter();
  setPlaceInstExtId();
  for (size_t iter = 1; iter <= ipVars_.maxIter; iter++) {
    updatePinInfo();
    const float error = createSparseMatrix();
    std::cout << "[InitialPlace]  Iter: " << iter << " "
              << "CG residual: " << error << " "
              << "HPWL: " << pbc_->hpwl() << std::endl;
  }
  updateCoordi();
  pbc_->updateWireLengthForceWA(1.0, 1.0);
  int cusp_major = CUSP_MAJOR_VERSION;
  int cusp_minor = CUSP_MINOR_VERSION;
  std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;
}

void GpuInitialPlace::placeInstsCenter()
{
  std::cout << "GpuInitialPlace : placeInstsCenter" << std::endl;
  const int centerX = pbc_->die().coreCx();
  const int centerY = pbc_->die().coreCy();
  pbc_->setInstCenterLocation(centerX, centerY);
}


struct FillEmptyCOOPair_functor
{
  COOPair* fixed_inst_force_vec_ptr;

  FillEmptyCOOPair_functor(
    COOPair* _fixed_inst_force_vec_ptr
  ) : fixed_inst_force_vec_ptr(_fixed_inst_force_vec_ptr)
    { }
 
  __host__ __device__
  void operator()(const COOPair& coo_pair) {
    fixed_inst_force_vec_ptr[coo_pair.idx].value = coo_pair.value;
  }
};

struct CreateSparseMatrix_functor 
{
  int* d_net_pin_idx;
  int* d_net_pin_ptr;
  GpuPin* d_pins_arr;
  GpuInstance* d_insts_arr;
  int* d_force_vec_ptr;
  COOPair* fixed_inst_force_vec_x_ptr;
  COOPair* fixed_inst_force_vec_y_ptr;
  COOTuple* ListX_ptr;
  COOTuple* ListY_ptr;
  int minDiffLength;
  int maxFanout;
  float netWeightScale;

  CreateSparseMatrix_functor(
    int* _d_net_pin_idx,
    int* _d_net_pin_ptr,
    GpuPin* _d_pins_arr,
    GpuInstance* _d_insts_arr,
    int* _d_force_vec_ptr,
    COOPair* _fixed_inst_force_vec_x_ptr,
    COOPair* _fixed_inst_force_vec_y_ptr,
    COOTuple* _listX_ptr,
    COOTuple* _listY_ptr,
    int _minDiffLength,
    int _maxFanout,
    float _netWeightScale) 
    : d_net_pin_idx(_d_net_pin_idx),
      d_net_pin_ptr(_d_net_pin_ptr),
      d_pins_arr(_d_pins_arr),
      d_insts_arr(_d_insts_arr),
      d_force_vec_ptr(_d_force_vec_ptr),
      fixed_inst_force_vec_x_ptr(_fixed_inst_force_vec_x_ptr),
      fixed_inst_force_vec_y_ptr(_fixed_inst_force_vec_y_ptr),
      ListX_ptr(_listX_ptr),
      ListY_ptr(_listY_ptr),
      minDiffLength(_minDiffLength),
      maxFanout(_maxFanout),
      netWeightScale(_netWeightScale)
    {  }

  __host__ __device__
  void operator()(GpuNet* net) {
    const int net_id = net->net_id_;
    const int num_pins =  d_net_pin_ptr[net_id + 1] - d_net_pin_ptr[net_id];
    if (num_pins <= 1 || num_pins >= maxFanout) {
      return;
    }  

    const float netWeight = netWeightScale / (num_pins - 1);
    const int pin_base = d_net_pin_ptr[net_id];
    const int d_force_vec_base = d_force_vec_ptr[net_id];
    int fixed_inst_force_vec_x_idx = d_force_vec_base;
    int ListX_idx = d_force_vec_base * 2;
    for (int idx1 = 1; idx1 < num_pins; idx1++) {
      GpuPin* pin1 = &d_pins_arr[d_net_pin_idx[pin_base + idx1]];
      for (int idx2 = 0; idx2 < idx1; idx2++) {
        GpuPin* pin2 = &d_pins_arr[d_net_pin_idx[pin_base + idx2]];
        // B2B modeling on min/maxX pins.
        if (pin1->isMinPinX() || pin1->isMaxPinX() || pin2->isMinPinX()
          || pin2->isMaxPinX()) {
          float diffX = abs(pin1->cx() - pin2->cx());
          float weightX = 0;
          if (diffX > minDiffLength) {
            weightX = netWeight / diffX;
          } else {
            weightX = netWeight / minDiffLength;
          }
          // both pin cames from instance
          if (pin1->isPlaceInstConnected() && pin2->isPlaceInstConnected()) {
            const int inst1 = d_insts_arr[pin1->inst_id_].extId();
            const int inst2 = d_insts_arr[pin2->inst_id_].extId();
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst1, weightX);
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst2, weightX);
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst2, -weightX);
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst1, -weightX);            
            float fixed_force = (pin1->cx() - d_insts_arr[pin1->inst_id_].cx())
                              - (pin2->cx() - d_insts_arr[pin2->inst_id_].cx());            
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst1, -weightX * fixed_force);
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst2, weightX * fixed_force);
          } 
          else if (!pin1->isPlaceInstConnected() && pin2->isPlaceInstConnected()) {
            // pin1 from IO port / pin2 from Instance 
            const int inst2 = d_insts_arr[pin2->inst_id_].extId();
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst2, weightX);
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst2, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst2, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(inst2, inst2, 0.0f);   
            float fixed_force = pin1->cx() - (pin2->cx() - d_insts_arr[inst2].cx());
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst2, weightX * fixed_force);
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst2, 0.0f);
          } else if (pin1->isPlaceInstConnected() && !pin2->isPlaceInstConnected()) {
            // pin1 from Instance / pin2 from IO port
            const int inst1 = d_insts_arr[pin1->inst_id_].extId();
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst1, weightX);
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst1, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst1, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(inst1, inst1, 0.0f);
            float fixed_force = pin2->cx() - (pin1->cx() - d_insts_arr[inst1].cx());
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst1, weightX * fixed_force);
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(inst1, 0.0f);
          } else {
            ListX_ptr[ListX_idx++] = COOTuple(0, 0, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(0, 0, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(0, 0, 0.0f);
            ListX_ptr[ListX_idx++] = COOTuple(0, 0, 0.0f);
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(0, 0.0f);
            fixed_inst_force_vec_x_ptr[fixed_inst_force_vec_x_idx++]
              = COOPair(0, 0.0f);
          }
        }

        int fixed_inst_force_vec_y_idx = d_force_vec_base;
        int ListY_idx = d_force_vec_base * 2;
        // B2B modeling on min/maxY pins.
        if (pin1->isMinPinY() || pin1->isMaxPinY() || pin2->isMinPinY()
         || pin2->isMaxPinY()) {
         int diffY = abs(pin1->cy() - pin2->cy());
         float weightY = 0;
         if (diffY > minDiffLength) {
           weightY = netWeight / diffY;
         } else {
           weightY = netWeight / minDiffLength;
         }
         // both pin cames from instance
         if (pin1->isPlaceInstConnected() && pin2->isPlaceInstConnected()) {
           const int inst1 = d_insts_arr[pin1->inst_id_].extId();
           const int inst2 = d_insts_arr[pin2->inst_id_].extId();
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst1, weightY);
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst2, weightY);
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst2, -weightY);
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst1, -weightY);            
           float fixed_force = (pin1->cy() - d_insts_arr[pin1->inst_id_].cy())
                             - (pin2->cy() - d_insts_arr[pin2->inst_id_].cy());            
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst1, -weightY * fixed_force);
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst2, weightY * fixed_force);
         } 
         else if (!pin1->isPlaceInstConnected() && pin2->isPlaceInstConnected()) {
           // pin1 from IO port / pin2 from Instance 
           const int inst2 = d_insts_arr[pin2->inst_id_].extId();
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst2, weightY);
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst2, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst2, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(inst2, inst2, 0.0f);   
           float fixed_force = pin1->cy() - (pin2->cy() - d_insts_arr[inst2].cy());
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst2, weightY * fixed_force);
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst2, 0.0f);
         } else if (pin1->isPlaceInstConnected() && !pin2->isPlaceInstConnected()) {
           // pin1 from Instance / pin2 from IO port
           const int inst1 = d_insts_arr[pin1->inst_id_].extId();
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst1, weightY);
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst1, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst1, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(inst1, inst1, 0.0f);
           float fixed_force = pin2->cy() - (pin1->cy() - d_insts_arr[inst1].cy());
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst1, weightY * fixed_force);
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(inst1, 0.0f);
         } else {
           ListY_ptr[ListY_idx++] = COOTuple(0, 0, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(0, 0, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(0, 0, 0.0f);
           ListY_ptr[ListY_idx++] = COOTuple(0, 0, 0.0f);
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(0, 0.0f);
           fixed_inst_force_vec_y_ptr[fixed_inst_force_vec_y_idx++]
             = COOPair(0, 0.0f);
          }
        } 
      }
    }    
  }
};
 

void mergeCOOPairs(cusp::array1d<COOPair, cusp::device_memory>& input_vec)
{
  // Sort COOPair objects by idx
  thrust::sort(thrust::device, input_vec.begin(), input_vec.end(), COOPairComp());
  // Allocate device memory for the merged output
  cusp::array1d<COOPair, cusp::device_memory> output(input_vec.size());
  // Merge COOPair objects with the same idx
  auto end = thrust::reduce_by_key(thrust::device, 
                                   input_vec.begin(), input_vec.end(), 
                                   input_vec.begin(),
                                   thrust::make_discard_iterator(), 
                                   output.begin(), 
                                   COOPairEqual(), 
                                   COOPairReduce());
  // Resize output to the actual size
  output.resize(thrust::get<1>(end) - output.begin());
  // remove zero value element
  cusp::array1d<COOPair, cusp::device_memory> output_nonzero(output.size());
  auto nonzero_end = thrust::copy_if(thrust::device, output.begin(), output.end(), 
                        output_nonzero.begin(), COOPairNonZero());
  // Resize output_nonzero to the actual size
  output_nonzero.resize(thrust::distance(output_nonzero.begin(), nonzero_end));                 
  // Overwrite input_vec with the merged output
  input_vec = output_nonzero;
}


void mergeCOOTuples(cusp::array1d<COOTuple, cusp::device_memory>& input_vec)
{
  // Sort COOPair objects by idx
  thrust::sort(thrust::device, input_vec.begin(), input_vec.end(), COOTupleComp());
  // Allocate device memory for the merged output
  cusp::array1d<COOTuple, cusp::device_memory> output(input_vec.size());
  // Merge COOPair objects with the same idx
  auto end = thrust::reduce_by_key(thrust::device, 
                                   input_vec.begin(), input_vec.end(), 
                                   input_vec.begin(),
                                   thrust::make_discard_iterator(), 
                                   output.begin(), 
                                   COOTupleEqual(), 
                                   COOTupleReduce());
  // Resize output to the actual size
  output.resize(thrust::get<1>(end) - output.begin());
  // remove zero value element
  cusp::array1d<COOTuple, cusp::device_memory> output_nonzero(output.size());
  auto nonzero_end = thrust::copy_if(thrust::device, output.begin(), output.end(), 
                             output_nonzero.begin(), COOTupleNonZero());
  // Resize output_nonzero to the actual size
  output_nonzero.resize(thrust::distance(output_nonzero.begin(), nonzero_end));                 
  // Overwrite input_vec with the merged output
  input_vec = output_nonzero;
}

struct TupleToTriple {
  __host__ __device__
  thrust::tuple<int, int, float> operator()(const COOTuple& t) const {
    return thrust::make_tuple(t.row_idx, t.col_idx, t.value);
  }
};

float SolveEquation(const cusp::array1d<COOTuple, cusp::device_memory>& tuple_list,
                    const cusp::array1d<COOPair, cusp::device_memory>& tuple_pair,
                    cusp::array1d<float, cusp::device_memory>& solu_vec,
                    int num_place_insts,
                    int maxSolverIter)
{
  // Create matrix
  // allocate output matrix
  cusp::coo_matrix<int, float, cusp::device_memory> matrix(num_place_insts, num_place_insts, tuple_list.size());
  // Create separate arrays for the row indices, column indices, and values
  cusp::array1d<int, cusp::device_memory> row_indices(tuple_list.size());
  cusp::array1d<int, cusp::device_memory> column_indices(tuple_list.size());
  cusp::array1d<float, cusp::device_memory> values(tuple_list.size());

  // Use transform to copy the elements of tuple_list into the separate arrays
  thrust::transform(tuple_list.begin(), 
    tuple_list.end(),
    thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin(), values.begin())),
    TupleToTriple());

  // Create the COO matrix
  matrix.row_indices = row_indices;
  matrix.column_indices = column_indices;
  matrix.values = values;

  // Convert the COOPair to b_vec
  cusp::array1d<float, cusp::device_memory> b_vec(num_place_insts, 0);
    thrust::transform(tuple_pair.begin(),
    tuple_pair.end(),
    b_vec.begin(),
    COOPairGetValue());

  // Solve the equation
  bool verbose = false;
  cusp::monitor<float> monitor(b_vec, maxSolverIter, 1e-6, 0, verbose);
  // set preconditioner (identity)
  //cusp::identity_operator<float, cusp::device_memory> M(matrix.num_cols, matrix.num_rows);
  cusp::precond::diagonal<float, cusp::device_memory> M(matrix);
  // solve the linear system A x = b
  cusp::krylov::bicgstab(matrix, solu_vec, b_vec, monitor, M);
  
  // check the residual error
  cusp::array1d<float, cusp::device_memory> Ax(num_place_insts);
  cusp::multiply(matrix, solu_vec, Ax);
  cusp::blas::axpy(b_vec, Ax, -1);

  // calculate the error
  return cusp::blas::nrm1(Ax) / cusp::blas::nrm1(b_vec);
  // copy the results back
  //cusp::array1d<float, cusp::host_memory> solu_vec_h(solu_vec);
  // Use std::copy to print the data
  //std::copy(solu_vec_h.begin(), solu_vec_h.end(), std::ostream_iterator<float>(std::cout, " "));
  //std::cout << std::endl;
}


// solve placeInstForceMatrixX_ * xcg_x_ = xcg_b_ and placeInstForceMatrixY_ *
// ycg_x_ = ycg_b_ eq.
float GpuInitialPlace::createSparseMatrix()
{ 
  std::cout << "GpuInitialPlace :: createSparseMatrix" << std::endl;
  const int num_place_insts = pbc_->getNumPlaceInsts();
  // create arrays for xcg_x and xcg_b_
  cusp::array1d<float, cusp::device_memory> inst_loc_vec_x(num_place_insts);
  cusp::array1d<float, cusp::device_memory> inst_loc_vec_y(num_place_insts);
  // We need to calculate the index range first
  std::vector<int> h_force_vec_ptr;
  h_force_vec_ptr.reserve(pbc_->getNumNets() + 1);
  h_force_vec_ptr.push_back(0);
  const std::vector<int>& h_net_pin_ptr = pbc_->hNetPinPtr();
  int num_elements =  0;
  for (int i = 0; i < pbc_->getNumNets(); i++) {
    const int num_pins = h_net_pin_ptr[i + 1] - h_net_pin_ptr[i];
    if (num_pins > 1 && num_pins < ipVars_.maxFanout) {
      num_elements += (num_pins - 1) * 2;
    }
    h_force_vec_ptr.push_back(num_elements);
  }

  thrust::device_vector<int> d_force_vec_ptr(num_elements);
  thrust::copy(h_force_vec_ptr.begin(), h_force_vec_ptr.end(), d_force_vec_ptr.begin());
  cusp::array1d<COOPair, cusp::device_memory> fixed_inst_force_vec_x(num_elements);
  cusp::array1d<COOPair, cusp::device_memory> fixed_inst_force_vec_y(num_elements); 
  cusp::array1d<COOTuple, cusp::device_memory> listX(num_elements * 2);
  cusp::array1d<COOTuple, cusp::device_memory> listY(num_elements * 2);

  // get the cooresponding raw pointer
  int* d_force_vec_ptr_raw = thrust::raw_pointer_cast(&d_force_vec_ptr[0]);
  COOPair* fixed_inst_force_vec_x_ptr = thrust::raw_pointer_cast(&fixed_inst_force_vec_x[0]);
  COOPair* fixed_inst_force_vec_y_ptr = thrust::raw_pointer_cast(&fixed_inst_force_vec_y[0]);
  COOTuple* listX_ptr = thrust::raw_pointer_cast(&listX[0]);
  COOTuple* listY_ptr = thrust::raw_pointer_cast(&listY[0]);
  
  // Now we are ready to create the sparse matrix
  thrust::device_ptr<GpuNet*> net_begin(pbc_->deviceNetsArray());
  thrust::device_ptr<GpuNet*> net_end = net_begin + pbc_->getNumNets();
  thrust::for_each(thrust::device,
    net_begin,
    net_end,
    CreateSparseMatrix_functor(pbc_->dNetPinIdx(),
                               pbc_->dNetPinPtr(),
                               pbc_->dPinsArray(),
                               pbc_->dInstsArray(),
                               d_force_vec_ptr_raw,
                               fixed_inst_force_vec_x_ptr,
                               fixed_inst_force_vec_y_ptr,
                               listX_ptr,
                               listY_ptr,
                               ipVars_.minDiffLength,
                               ipVars_.maxFanout,
                               ipVars_.netWeightScale));

  // merge COOPairs
  mergeCOOPairs(fixed_inst_force_vec_x);
  mergeCOOPairs(fixed_inst_force_vec_y);

  if (fixed_inst_force_vec_x.size() != num_place_insts || 
      fixed_inst_force_vec_y.size() != num_place_insts) {
    thrust::device_vector<int> index_vec(num_place_insts);
    thrust::sequence(index_vec.begin(), index_vec.end());
    if (fixed_inst_force_vec_x.size() != num_place_insts) {
      cusp::array1d<COOPair, cusp::device_memory> fixed_inst_force_vec_x_temp(num_place_insts);
      thrust::transform(index_vec.begin(),
                        index_vec.end(),
                        fixed_inst_force_vec_x_temp.begin(),
                        [=] __device__ (int idx) {
                          return COOPair(idx, 0);
                        });

      COOPair* fixed_inst_force_vec_x_ptr_temp = thrust::raw_pointer_cast(&fixed_inst_force_vec_x_temp[0]);
      COOPair* fixed_inst_force_vec_x_ptr_raw = thrust::raw_pointer_cast(&fixed_inst_force_vec_x[0]);
      thrust::device_ptr<COOPair> dev_ptr(fixed_inst_force_vec_x_ptr_raw);
      thrust::for_each(thrust::device,
        dev_ptr,
        dev_ptr + fixed_inst_force_vec_x.size(),
        FillEmptyCOOPair_functor(fixed_inst_force_vec_x_ptr_temp));
      fixed_inst_force_vec_x = fixed_inst_force_vec_x_temp;
    }   
    
    if (fixed_inst_force_vec_y.size() != num_place_insts) {
      cusp::array1d<COOPair, cusp::device_memory> fixed_inst_force_vec_y_temp(num_place_insts);
      thrust::transform(index_vec.begin(),
                        index_vec.end(),
                        fixed_inst_force_vec_y_temp.begin(),
                        [=] __device__ (int idx) {
                          return COOPair(idx, 0);
                        });

      COOPair* fixed_inst_force_vec_y_ptr_temp = thrust::raw_pointer_cast(&fixed_inst_force_vec_y_temp[0]);
      COOPair* fixed_inst_force_vec_y_ptr_raw = thrust::raw_pointer_cast(&fixed_inst_force_vec_y[0]);
      thrust::device_ptr<COOPair> dev_ptr(fixed_inst_force_vec_y_ptr_raw);
      thrust::for_each(thrust::device,
        dev_ptr,
        dev_ptr + fixed_inst_force_vec_y.size(),
        FillEmptyCOOPair_functor(fixed_inst_force_vec_y_ptr_temp));
      fixed_inst_force_vec_y = fixed_inst_force_vec_y_temp;
    }   
  }
  
  // merge COOTuples
  mergeCOOTuples(listX);
  mergeCOOTuples(listY);

  cusp::array1d<float, cusp::device_memory> instLocVecX(num_place_insts, 0);
  cusp::array1d<float, cusp::device_memory> instLocVecY(num_place_insts, 0);
  float error_x = SolveEquation(listX, fixed_inst_force_vec_x, instLocVecX, num_place_insts, ipVars_.maxSolverIter);
  float error_y = SolveEquation(listY, fixed_inst_force_vec_y, instLocVecY, num_place_insts, ipVars_.maxSolverIter);

  // convert the float to inst
  cusp::array1d<int, cusp::device_memory> instLocVecX_int(num_place_insts);
  cusp::array1d<int, cusp::device_memory> instLocVecY_int(num_place_insts);
  thrust::transform(instLocVecX.begin(), instLocVecX.end(), instLocVecX_int.begin(), FloatToIntFunctor());
  thrust::transform(instLocVecY.begin(), instLocVecY.end(), instLocVecY_int.begin(), FloatToIntFunctor());
  
  pbc_->setPlaceInstLocation(thrust::raw_pointer_cast(instLocVecX_int.data()),
                             thrust::raw_pointer_cast(instLocVecY_int.data()));
  return max(error_x, error_y);
}

struct unsetPin_functor
{
  __host__ __device__
  void operator()(GpuPin* pin) const {
    pin->unsetMinPinX();
    pin->unsetMinPinY();
    pin->unsetMaxPinX();
    pin->unsetMaxPinY();
  }
};

// B2B net modeling
struct UpdateB2BInfo_functor
{
  int* d_net_pin_idx;
  int* d_net_pin_ptr;
  GpuPin* d_pins_arr;
  
  UpdateB2BInfo_functor(
    int* _d_net_pin_idx,
    int* _d_net_pin_ptr,
    GpuPin* _d_pins_arr) 
    : d_net_pin_idx(_d_net_pin_idx),
      d_net_pin_ptr(_d_net_pin_ptr),
      d_pins_arr(_d_pins_arr) {  }

  
  __host__ __device__ 
  void operator()(GpuNet* net) const {
    int pinMinX_id = -1;
    int pinMinY_id = -1;
    int pinMaxX_id = -1;
    int pinMaxY_id = -1;
    int lx = INT_MAX;
    int ly = INT_MAX;
    int ux = INT_MIN;
    int uy = INT_MIN;
    int net_id = net->net_id_;
    int num_pins = d_net_pin_ptr[net_id + 1] - d_net_pin_ptr[net_id];
    if (num_pins <= 1) {
      return;
    }
    for (int i = d_net_pin_ptr[net_id];
         i < d_net_pin_ptr[net_id + 1];
         i++) {
      int idx = d_net_pin_idx[i];
      int cx = d_pins_arr[idx].cx();
      int cy = d_pins_arr[idx].cy();
      if (lx > cx) {
        lx = cx;
        pinMinX_id = idx;
      }
  
      if (ux < cx) {
        ux = cx;
        pinMaxX_id = idx;
      }
  
      if (ly > cy) {
        ly = cy;
        pinMinY_id = idx;
      }
  
      if (uy < cy) {
        uy = cy;
        pinMaxY_id = i;
      }
    }
    
    if (pinMinX_id == -1 || pinMinY_id == -1 || pinMaxX_id == -1 || pinMaxY_id == -1) {
      return;
    } else {
      d_pins_arr[pinMinX_id].setMinPinX();
      d_pins_arr[pinMinY_id].setMinPinY();
      d_pins_arr[pinMaxX_id].setMaxPinX();
      d_pins_arr[pinMaxY_id].setMaxPinY(); 
    }
  }
};

void GpuInitialPlace::updatePinInfo()
{
  std::cout << "GpuInitialPlace :: updatePinInfo" << std::endl;
  thrust::device_ptr<GpuPin*> pin_begin(pbc_->devicePinsArray());
  thrust::device_ptr<GpuPin*> pin_end = 
              pin_begin + pbc_->getNumPins();
  thrust::for_each(thrust::device, 
    pin_begin,
    pin_end,
    [=] __device__ (GpuPin* pin) {
      pin->unsetMinPinX();
      pin->unsetMinPinY();
      pin->unsetMaxPinX();
      pin->unsetMaxPinY();
    }
  );

  thrust::device_ptr<GpuNet*> net_begin(pbc_->deviceNetsArray());
  thrust::device_ptr<GpuNet*> net_end = 
                    net_begin +  pbc_->getNumNets();
  thrust::for_each(thrust::device,
    net_begin,
    net_end,
    UpdateB2BInfo_functor(pbc_->dNetPinIdx(),
                          pbc_->dNetPinPtr(),
                          pbc_->dPinsArray()));
}


void GpuInitialPlace::setPlaceInstExtId()
{
  std::cout << "GpuInitialPlace :: setPlaceInstExtId" << std::endl;
  // reset ExtId for all instances
  thrust::device_ptr<GpuInstance*> inst_begin(pbc_->deviceInstsArray());
  thrust::device_ptr<GpuInstance*> inst_end = inst_begin + pbc_->getNumInsts();
  thrust::for_each(thrust::device,
    inst_begin,
    inst_end,
    [=] __device__ (GpuInstance* inst) {
        inst->setExtId(INT_MAX);
    }
  );
  // set the ExtId for all place instances
  thrust::device_ptr<GpuInstance*> place_inst_begin(pbc_->devicePlaceInstsArray());
  thrust::device_ptr<GpuInstance*> place_inst_end = 
        place_inst_begin + pbc_->getNumPlaceInsts();
  thrust::for_each(thrust::device,
    place_inst_begin,
    place_inst_end,
    [=] __device__ (GpuInstance* inst) {
        inst->setExtId(inst->inst_id_);
    }
  );
}

void GpuInitialPlace::updateCoordi()
{
  std::cout << "updateCoordi" << std::endl;
  // copy from device to host
  pbc_->SyncInstD2H();
  pbc_->updateDB();
}

void GpuInitialPlace::reset()
{
  pbc_ = nullptr;
  ipVars_.reset();
}

}  // namespace gpl2
