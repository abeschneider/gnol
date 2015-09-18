//
//  module.cpp
//  rnn
//
//  Created by Abraham Schneider on 9/3/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "module.hpp"

using namespace gnol;

Module::Module(size_t input_size, size_t output_size):
    input_size(input_size),
    output_size(output_size) {}


GradientModule::GradientModule(size_t input_size, size_t output_size):
    Module(input_size, output_size)
{
    if (input_size.dims() == 1)
        grad_input.resize(input_size[0], 1);
    else
        grad_input.resize(input_size[0], input_size[1]);
}
