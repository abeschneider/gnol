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
#include <utility>

#include <initializer_list>
#include <boost/range.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "utility.hpp"

namespace gnol {
    using namespace arma;
        
    class Module {
    protected:
        variable<matrix_t> output;
        size_t input_size, output_size;
    public:
        Module(size_t input_size, size_t output_size);
        Module(ssize_t<1> input_size, ssize_t<1> output_size);
        Module(ssize_t<2> input_size, ssize_t<2> output_size);
        
        size_t get_input_size() const { return input_size; }
        size_t get_output_size() const { return output_size; }
        variable<matrix_t> &get_output() { return output; }
        
        virtual matrix_t &forward(const matrix_t &input) = 0;
        virtual parameter_list flatten_parameters() = 0;
    };
    
    class GradientModule: public Module {
    protected:
        matrix_t grad_input;
    public:
        GradientModule(size_t input_size, size_t output_size);
        
        matrix_t &get_grad_input() { return grad_input; }
        
        virtual void clear() { grad_input.zeros(); }
        virtual matrix_t &backward(const matrix_t &input, const matrix_t &grad_output) = 0;
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
        
        matrix_t &forward(const matrix_t &input) {
            op(params, input, *output);
            return *output;
        }
        
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output) {
            grad(params, grad_params, input, grad_output, grad_input);
            return grad_input;
        }
        
        ParamT &get_params() { return params; }
        GradParamT &get_grad_params() { return grad_params; }
        
        virtual parameter_list flatten_parameters() { return params.flatten(); }
        virtual parameter_list flatten_deriv_parameters() { return grad_params.flatten(); }
        
        virtual void clear() {
            grad_input.zeros();
            grad_params.clear();
        }
    };
                
    template <typename T, typename...Args>
    std::shared_ptr<T> make_module(Args...args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
}

#endif /* defined(__rnn__module__) */
