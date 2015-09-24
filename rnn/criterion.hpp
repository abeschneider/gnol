//
//  criterion.h
//  rnn
//
//  Created by Abe Schneider on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__criterion__
#define __rnn__criterion__

#include <armadillo>

#include "utility.hpp"
#include "module.hpp"

using namespace arma;

namespace gnol {
    class Criterion {
    public:
        virtual real_t forward(const vector_t &input, const vector_t &target) = 0;
        virtual matrix_t &backward(const vector_t &input, const vector_t &target) = 0;
    };
    
    struct L2Op {
        real_t operator ()(const vector_t &input, const vector_t &target) {
            return 0.5*sum(pow(input - target, 2));
        }
    };
    
    struct L2Gradient {
        vector_t operator ()(const vector_t &input, const vector_t &target) {
            return input - target;
        }
    };
    
    class L2Loss: public Criterion {
    private:
        L2Op op;
        L2Gradient grad;
        vector_t grad_input;
    public:
        real_t forward(const vector_t &input, const vector_t &target) {
//            auto result = 0.5*sum(pow(input - target, 2));
//            return std::move(result);
            return op(input, target);
        }
        
        matrix_t &backward(const vector_t &input, const vector_t &target) {
//            grad_input = (input - target);
//            return grad_input;
            grad_input = grad(input, target);
            return grad_input;
        }
    };
    
    template <typename OpT, typename GradT>
    class CriterionModule: public GradientModule {
    protected:
        OpT op;
        GradT grad;
        variable<matrix_t> target;
    public:
        CriterionModule(variable<matrix_t> &target):
            GradientModule(target->n_rows, target->n_rows),
            target(target) {}
        
        CriterionModule(variable<matrix_t> &&target):
            GradientModule(target->n_rows, target->n_rows),
            target(target) {}
        
        matrix_t &forward(const matrix_t &input) {
            *output = op(input, *target);
            return *output;
        }
        
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output) {
            grad_input = grad(input, *target);
            return grad_input;
        }
        
        parameter_list flatten_parameters() { return empty_parameter_list; }
        parameter_list flatten_deriv_parameters() { return empty_parameter_list; }
        
        virtual void clear() { grad_input.zeros(); }
    };
}

#endif /* defined(__rnn__criterion__) */
