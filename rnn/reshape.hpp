//
//  reshape.h
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__reshape__
#define __rnn__reshape__

#include "module.hpp"

namespace gnol {
    template <std::size_t D>
    class ReshapeModule: public GradientModule {};
    
    template <>
    class ReshapeModule<2>: public GradientModule {
    protected:
        fmat reshaped_input;
        fmat reshaped_grad_input;
    public:
        ReshapeModule(ssize_t<2> input_size, ssize_t<2> output_size):
            GradientModule(input_size, output_size) {}
        
        fmat &forward(const fmat &input) {
            output = input.submat(0, 0, input.n_rows-1, input.n_cols-1);
            output.reshape(output_size[0], output_size[1]);
            
            return output;
        }
        
        fmat &backward(const fmat &input, const fmat &grad_output) {
            grad_input = grad_output.submat(0, 0, grad_output.n_rows - 1, grad_output.n_cols - 1);
            grad_input.reshape(input_size[0], input_size[1]);
            return grad_input;
        }
        
        parameter_list flatten_parameters() {
            return parameter_list();
        }
        
        parameter_list flatten_deriv_parameters() {
            return parameter_list();
        }
    };
}
#endif /* defined(__rnn__reshape__) */
