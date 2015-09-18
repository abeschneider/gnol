//
//  module.h
//  rnn
//
//  Created by Abe Schneider on 9/3/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__module__
#define __rnn__module__

#include <armadillo>

#include <list>
#include <initializer_list>
#include <utility>
#include <boost/range.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "utility.hpp"

namespace gnol {
    using namespace arma;
    
    class Module {
    protected:
        fmat output;
        size_t input_size, output_size;
    public:
        Module(size_t input_size, size_t output_size);
        
        size_t get_input_size() const { return input_size; }
        size_t get_output_size() const { return output_size; }
        fmat &get_output() { return output; }
        
        virtual fmat &forward(const fmat &input) = 0;
        virtual parameter_list flatten_parameters() = 0;
    };
    
    class GradientModule: public Module {
    protected:
        fmat grad_input;
    public:
        GradientModule(size_t input_size, size_t output_size);
        
        fmat &get_grad_input() { return grad_input; }
        
        virtual void clear() { grad_input.zeros(); }
        virtual fmat &backward(const fmat &input, const fmat &grad_output) = 0;
        virtual parameter_list flatten_deriv_parameters() = 0;
    };
        
    template <typename OpT, typename ParamT, typename GradOpT, typename GradParamT>
    class ParameterizedModule: public GradientModule {
    protected:
        OpT op;
        ParamT params;
        GradOpT grad;
        GradParamT grad_params;
    protected:
        ParameterizedModule(ParamT &&params,
                            GradParamT &&grad_params,
                            ssize_t<2> size):
            GradientModule(size[0], size[1]),
            params(std::move(params)),
            grad_params(std::move(grad_params)) {}

    public:
        ParameterizedModule(ssize_t<2> size):
            GradientModule(size[0], size[1]),
            params(size),
            grad_params(size) {}
        
        ParameterizedModule(const ParamT &params,
                            const GradParamT &grad_params):
            GradientModule(params.weight->n_rows, params.weight->n_cols),
            params(params),
            grad_params(grad_params) {}
        
        fmat &forward(const fmat &input) {
            op(params, input, output);
            return output;
        }
        
        fmat &backward(const fmat &input, const fmat &grad_output) {
            grad(params, grad_params, input, grad_output, grad_input);
            return grad_input;
        }
        
        ParamT &get_params() { return params; }
        GradParamT &get_grad_params() { return grad_params; }
        
        virtual parameter_list flatten_parameters() { return params.flatten(); }
        virtual parameter_list flatten_deriv_parameters() { return grad_params.flatten(); }
    };
                
    template <typename T, typename...Args>
    std::shared_ptr<T> make_module(Args...args) {
        return std::make_shared<T>(args...);
    }
}

#endif /* defined(__rnn__module__) */
