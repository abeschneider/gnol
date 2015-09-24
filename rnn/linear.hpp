//
//  linear.hpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__linear__
#define __rnn__linear__

#include "module.hpp"

#pragma GCC visibility push(default)

namespace gnol {
    struct LinearParams {
        variable<matrix_t> weight;
        variable<vector_t> bias;
        
        LinearParams(LinearParams &&params):
            weight(params.weight),
            bias(params.bias) {}
        
        LinearParams(ssize_t<2> size):
            weight(size),
            bias(size[1])
        {
            weight->randu();
            bias->randu();
        }
        
        LinearParams(variable<matrix_t> &w, variable<vector_t> &b):
            weight(w),
            bias(b)
        {
            bias->randu();
        }
        
        LinearParams(variable<matrix_t> &&w, variable<vector_t> &&b):
            weight(w),
            bias(b)
        {
            bias->randu();
        }
        
        LinearParams(matrix_t &w, vector_t &b):
            weight(w),
            bias(b)
        {
            bias->randu();
        }
        
        
        LinearParams(variable<matrix_t> &w):
            weight(w),
            bias(w->n_rows)
        {
            bias->randu();
        }
        
        LinearParams(variable<matrix_t> &&w):
            weight(w),
            bias(w->n_rows)
        {
            bias->randu();
        }
        
        void resize(ssize_t<2> size);
        parameter_list flatten();
    };
    
    struct LinearGradParams {
        variable<matrix_t> weight;
        variable<vector_t> bias;
        
        LinearGradParams(ssize_t<2> size):
            weight(size),
            bias(size[1])
        {
            clear();
        }
        
        LinearGradParams(variable<matrix_t> &weight, variable<vector_t> &bias):
            weight(weight),
            bias(bias)
        {
            clear();
        }
        
        LinearGradParams(variable<matrix_t> &&weight, variable<vector_t> &&bias):
            weight(weight),
            bias(bias)
        {
            clear();
        }
        
        LinearGradParams(variable<matrix_t> &grad_weight):
            weight(grad_weight),
            bias(grad_weight->n_rows)
        {
            clear();
        }
        
        LinearGradParams(variable<matrix_t> &&grad_weight):
            weight(grad_weight),
            bias(grad_weight->n_rows)
        {
            clear();
        }
        
        void clear();
        parameter_list flatten();
    };
    
    struct LinearOp {
        void operator ()(LinearParams &params, const matrix_t &input, matrix_t &output) {
            output = params.weight->t()*input + *params.bias;
        }
    };
    
    struct LinearGradient {
        void operator ()(LinearParams &params, LinearGradParams &gparams, const matrix_t &input, const matrix_t &grad_output, matrix_t &grad_input) {
            *gparams.weight += (grad_output*input.t()).t();
            *gparams.bias += grad_output;
            grad_input += *(params.weight)*grad_output;
        }
    };
    
//    typedef ParameterizedModule<LinearOp, LinearParams, LinearGradient, LinearGradParams> LinearModule;
    struct LinearModule: public ParameterizedModule<LinearOp, LinearParams, LinearGradient, LinearGradParams> {
    public:
        LinearModule(ssize_t<2> size):
            ParameterizedModule(size)
        {
//            output.resize(params.weight->n_rows);
        }
        
        LinearModule(LinearParams &&params, LinearGradParams &&grad_params);        
        LinearModule(variable<matrix_t> &weight, variable<matrix_t> &grad_weight);
    };
    
    struct TransposedLinearOp {
        void operator ()(LinearParams &params, const matrix_t &input, matrix_t &output) {
            output = (*params.weight)*input + *params.bias;
        }
    };
    
    struct TransposedLinearGradient {
        void operator ()(LinearParams &params, LinearGradParams &gparams, const matrix_t &input, const matrix_t &grad_output, matrix_t &grad_input) {
            *gparams.weight += (input*grad_output.t()).t(); // TODO: simplify
            *gparams.bias += grad_output;
            grad_input += params.weight->t()*grad_output;
        }
    };
    
    class TransposedLinearModule: public ParameterizedModule<TransposedLinearOp, LinearParams, TransposedLinearGradient, LinearGradParams> {
    public:
        TransposedLinearModule(LinearParams &&params,
                               LinearGradParams &&grad_params):
            ParameterizedModule(LinearParams(std::move(params.weight)),
                                LinearGradParams(std::move(grad_params.weight)),
                                {params.weight->n_cols, params.weight->n_rows}) {}
    };
}

#pragma GCC visibility pop

#endif /* defined(__rnn__linear__) */
