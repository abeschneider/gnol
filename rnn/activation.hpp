//
//  activation.hpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef rnn_activation_hpp
#define rnn_activation_hpp

#include "module.hpp"

namespace gnol {
    class SigmoidModule: public GradientModule {
    public:
        SigmoidModule(size_t size):
            GradientModule(size, size) {}
        
        SigmoidModule(ssize_t<1> size):
            GradientModule(size, size) {}
        
        matrix_t &forward(const matrix_t &input) {
            *output = 1.0 / (1.0 + exp(-input));
            return *output;
        }
        
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output) {
            grad_input += grad_output % (1.0 - *output) % *output;
            return grad_input;
        }
        
        virtual parameter_list flatten_parameters() { return parameter_list(); }
        virtual parameter_list flatten_deriv_parameters() { return parameter_list(); }
    };
}

#endif
