//
//  op.cpp
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "op.hpp"

using namespace weight;

LinearOp::LinearOp(ssize_t<1> input_size, ssize_t<1> output_size):
    weight(input_size[0], output_size[0]),
    bias(output_size[0])
{
    weight.randu();
    bias.randu();
}

parameter_list LinearOp::get_parameters() {
    parameter_list params = {
        boost::make_iterator_range(weight.begin(), weight.end()),
        boost::make_iterator_range(bias.begin(), bias.end())
    };
    
    return std::move(params);
}

parameter_list SigmoidOp::get_parameters() {
    return parameter_list();
}
