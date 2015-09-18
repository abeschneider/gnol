//
//  grad.cpp
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "gradient.hpp"

using namespace weight;

LinearGradient::LinearGradient(LinearOp &op):
    op(op),
    grad_weight(op.input_size(), op.output_size()),
    grad_bias(op.output_size())
{
    reset();
}

void LinearGradient::reset() {
    grad_weight.zeros();
    grad_bias.zeros();
}

void LinearGradient::operator ()(const fvec &input,
                                 const fvec &output,
                                 const fvec &grad_output,
                                 fvec &grad_input)
{
    grad_weight += input*grad_output.t();
    grad_bias += grad_output;
    
    grad_input += op.weight*grad_output;
}

void LinearGradient::operator ()(const fmat &input,
                                 const fmat &output,
                                 const fmat &grad_output,
                                 fmat &grad_input)
{
    grad_weight += input*grad_output.t();
    grad_bias += grad_output;
    
    grad_input += op.weight*grad_output;
}

parameter_list LinearGradient::get_deriv_parameters() {
    parameter_list params = {
        boost::make_iterator_range(grad_weight.begin(), grad_weight.end()),
        boost::make_iterator_range(grad_bias.begin(), grad_bias.end())
    };
    
    return std::move(params);
}


TransposeGradient::TransposeGradient(TransposeOp &op):
    op(op)
{
    grad_weight.resize(op.linear->output_size(), op.linear->input_size());
    grad_bias.resize(op.linear->input_size());
    reset();
}

void TransposeGradient::reset() {
    grad_weight.zeros();
    grad_bias.zeros();
}

void TransposeGradient::operator ()(const fvec &input,
                                    const fvec &output,
                                    const fvec &grad_output,
                                    fvec &grad_input)
{
    grad_weight += input*grad_output.t();
    grad_bias += grad_output;
    
    grad_input += op.linear->weight.t()*grad_output;
}

void TransposeGradient::operator ()(const fmat &input,
                                    const fmat &output,
                                    const fmat &grad_output,
                                    fmat &grad_input)
{
    grad_weight += input*grad_output.t();
    grad_bias += grad_output;
    
    grad_input += op.linear->weight.t()*grad_output;
}

parameter_list TransposeGradient::get_deriv_parameters() {
    parameter_list params = {
        boost::make_iterator_range(grad_weight.begin(), grad_weight.end()),
        boost::make_iterator_range(grad_bias.begin(), grad_bias.end())
    };
    
    return std::move(params);
}

SigmoidGradient::SigmoidGradient(SigmoidOp &op):
    op(op) {}

void SigmoidGradient::operator ()(const fmat &input,
                                  const fmat &output,
                                  const fmat &grad_output,
                                  fmat &grad_input)
{
    grad_input += grad_output % (1.0 - output) % output;
}

