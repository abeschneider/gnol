//
//  convolve.h
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__convolve__
#define __rnn__convolve__

#include "module.hpp"

namespace gnol {
    void convolve2d(const fmat &input, const fmat &kernel, fmat &output);

    struct Convolve2DParams {
        variable<fmat> kernel;
        
        parameter_list flatten() {
            parameter_list params = {
                boost::make_iterator_range(kernel->begin(), kernel->end()),
            };
            
            return std::move(params);
        }
    };

    struct Convolve2DGradParams {
        variable<fmat> kernel;
        
        Convolve2DGradParams(ssize_t<2> size):
        kernel(size)
        {
            clear();
        }
        
        Convolve2DGradParams(variable<fmat> &kernel):
        kernel(kernel)
        {
            clear();
        }
        
        Convolve2DGradParams(variable<fmat> &&kernel):
        kernel(kernel)
        {
            clear();
        }
        
        void clear() {
            kernel->zeros();
        }
        
        parameter_list flatten() {
            parameter_list params = {
                boost::make_iterator_range(kernel->begin(), kernel->end()),
            };
            
            return std::move(params);
        }
    };

    struct Convolve2DTransform {
        void operator()(Convolve2DParams &params, const fmat &input, fmat &output) {
            convolve2d(input, *params.kernel, output);
        }
    };

    struct Convolve2DGradient {
        void operator ()(Convolve2DParams &params, Convolve2DGradParams &gparams, const fmat &input, const fmat &grad_output, fmat &grad_input) {
            //            *gparams.weight += input*grad_output.t();
            //            *gparams.bias += grad_output;
            //            grad_input += *(params.weight)*grad_output;
            
            //            gparams.kernel +=
            //            convolve2d(input, (*params.kernel).t(), grad_input);
        }
    };
}

#endif /* defined(__rnn__convolve__) */
