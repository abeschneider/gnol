//
//  linear.cpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "linear.hpp"

using namespace gnol;

LinearModule::LinearModule(LinearParams &&params, LinearGradParams &&grad_params):
    ParameterizedModule(
        LinearParams(std::move(params.weight),
                     make_vector(params.weight->n_rows)),
        LinearGradParams(std::move(grad_params.weight),
                         make_vector(grad_params.weight->n_rows)),
        {params.weight->n_cols, params.weight->n_rows}) {}


LinearModule::LinearModule(variable<matrix_t> &weight, variable<matrix_t> &grad_weight):
    ParameterizedModule(
        LinearParams(weight),
        LinearGradParams(grad_weight),
        {weight->n_cols, weight->n_rows}) {}

void LinearParams::resize(ssize_t<2> size) {
    weight->resize(size[0], size[1]);
    bias->resize(size[1]);
}

parameter_list LinearParams::flatten() {
    parameter_list params = {
        boost::make_iterator_range(weight->begin(), weight->end()),
        boost::make_iterator_range(bias->begin(), bias->end())
    };
    
    return std::move(params);
}

void LinearGradParams::clear() {
    weight->zeros();
    bias->zeros();
}

parameter_list LinearGradParams::flatten() {
    parameter_list params = {
        boost::make_iterator_range(weight->begin(), weight->end()),
        boost::make_iterator_range(bias->begin(), bias->end())
    };
    
    return std::move(params);
}
