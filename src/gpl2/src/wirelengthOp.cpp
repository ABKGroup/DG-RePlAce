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

#pragma once

#include <memory>
#include <ctime>

#include "placerObjects.h"
#include "placerBase.h"
#include "wirelengthOp.h"

namespace gpl2 {   

//////////////////////////////////////////////////////////////
// Class WirelengthOp

WirelengthOp::WirelengthOp()
  : pbc_(nullptr),
    logger_(nullptr),
    numInsts_(0),
    numPins_(0),
    numNets_(0),
    dInstPinIdxPtr_(nullptr),
    dInstPinPosPtr_(nullptr),
    dPinInstIdPtr_(nullptr),
    dNetPinIdxPtr_(nullptr),
    dNetPinPosPtr_(nullptr),
    dPinNetIdPtr_(nullptr),
    // pin information
    dPinXPtr_(nullptr),
    dPinYPtr_(nullptr),
    dPinOffsetXPtr_(nullptr),
    dPinOffsetYPtr_(nullptr),
    dPinGradXPtr_(nullptr),
    dPinGradYPtr_(nullptr),
    DREAMPlaceFlag_(false), 
    // net information
    dNetWidthPtr_(nullptr),
    dNetHeightPtr_(nullptr),
    runtime_(0.0)
    {  }


WirelengthOp::WirelengthOp(PlacerBaseCommon* pbc)
    : WirelengthOp()
{
  pbc_ = pbc;
  logger_ = pbc_->logger();
  logger_->report("[WirelengthOp] Start Initialization.");

  // placeable instances + fixed instances
  numInsts_ = pbc_->numInsts();
  numPlaceInsts_ = pbc_->numPlaceInsts();
  numPins_ = pbc_->numPins();
  numNets_ = pbc_->numNets();

  initCUDAKernel();
  logger_->report("[WirelengthOp] Initialization Succeed.");
}

WirelengthOp::~WirelengthOp()
{
  freeCUDAKernel();
}

void WirelengthOp::enableDREAMPlaceFlag()
{
  DREAMPlaceFlag_ = true;
}

}