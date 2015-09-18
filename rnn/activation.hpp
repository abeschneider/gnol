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
        SigmoidModule(ssize_t<1> size):
        GradientModule(size, size) {}
        
        fmat &forward(const fmat &input) {
            output = 1.0 / (1.0 + exp(-input));
            return output;
        }
        
        fmat &backward(const fmat &input, const fmat &grad_output) {
            grad_input += grad_output % (1.0 - output) % output;
            return grad_input;
        }
        
        virtual parameter_list flatten_parameters() { return parameter_list(); }
        virtual parameter_list flatten_deriv_parameters() { return parameter_list(); }
    };
}

#endif
