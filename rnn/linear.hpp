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
        variable<fmat> weight;
        variable<fvec> bias;
        
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
        
        LinearParams(variable<fmat> &w, variable<fvec> &b):
            weight(w),
            bias(b) {}
        
        LinearParams(variable<fmat> &&w, variable<fvec> &&b):
            weight(w),
            bias(b) {}
        
        LinearParams(fmat &w, fvec &b):
            weight(w),
            bias(b) {}
        
        
        LinearParams(variable<fmat> &w):
            weight(w),
            bias(w->n_rows)
        {
            bias->randu();
        }
        
        LinearParams(variable<fmat> &&w):
            weight(w),
            bias(w->n_rows)
        {
            bias->randu();
        }
        
        void resize(ssize_t<2> size);
        parameter_list flatten();
    };
    
    struct LinearGradParams {
        variable<fmat> weight;
        variable<fvec> bias;
        
        LinearGradParams(ssize_t<2> size):
            weight(size),
            bias(size[1])
        {
            clear();
        }
        
        LinearGradParams(variable<fmat> &weight, variable<fvec> &bias):
            weight(weight),
            bias(bias)
        {
            clear();
        }
        
        LinearGradParams(variable<fmat> &&weight, variable<fvec> &&bias):
            weight(weight),
            bias(bias)
        {
            clear();
        }
        
        LinearGradParams(variable<fmat> &grad_weight):
            weight(grad_weight),
            bias(grad_weight->n_rows)
        {
            clear();
        }
        
        LinearGradParams(variable<fmat> &&grad_weight):
            weight(grad_weight),
            bias(grad_weight->n_rows)
        {
            clear();
        }
        
        void clear();
        parameter_list flatten();
    };
    
    struct LinearOp {
        void operator ()(LinearParams &params, const fmat &input, fmat &output) {
            output = params.weight->t()*input + *params.bias;
        }
    };
    
    struct LinearGradient {
        void operator ()(LinearParams &params, LinearGradParams &gparams, const fmat &input, const fmat &grad_output, fmat &grad_input) {
            *gparams.weight += input*grad_output.t();
            *gparams.bias += grad_output;
            grad_input += *(params.weight)*grad_output;
        }
    };
    
    typedef ParameterizedModule<LinearOp, LinearParams, LinearGradient, LinearGradParams> LinearModule;
    
    struct TransposedLinearOp {
        void operator ()(LinearParams &params, const fmat &input, fmat &output) {
            output = (*params.weight)*input + *params.bias;
        }
    };
    
    struct TransposedLinearGradient {
        void operator ()(LinearParams &params, LinearGradParams &gparams, const fmat &input, const fmat &grad_output, fmat &grad_input) {
            *gparams.weight += (input*grad_output.t()).t(); // TODO: simplify
            *gparams.bias += grad_output;
            grad_input += params.weight->t()*grad_output;
        }
    };
    
    class TransposedLinearModule: public ParameterizedModule<TransposedLinearOp, LinearParams, TransposedLinearGradient, LinearGradParams> {
    public:
        TransposedLinearModule(LinearParams &&params,
                               LinearGradParams &&grad_params):
            ParameterizedModule(LinearParams(std::move(params.weight),
                                             make_vector(params.weight->n_rows)),
                                LinearGradParams(std::move(grad_params.weight),
                                                 make_vector(grad_params.weight->n_rows)),
                                {params.weight->n_cols, params.weight->n_rows}) {}
    };
}

#pragma GCC visibility pop

#endif /* defined(__rnn__linear__) */
