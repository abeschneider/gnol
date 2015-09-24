//
//  check_gradient.cpp
//  rnn
//
//  Created by Abe Schneider on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "check_gradient.hpp"
#include "criterion.hpp"

//using namespace gnol;
namespace gnol {
    std::list<real_t> check_gradient(std::function<real_t (const vector_t &)> fn,
                                    GradientModule &mod,
                                    const vector_t &input,
                                    real_t eps)
    {
        std::list<real_t> result;
        
        // calculate gradients
        fn(input);

        // flatten out parameters so we can iterate over them below
        parameter_list params = mod.flatten_parameters();
        parameter_list dparams = mod.flatten_deriv_parameters();
        
        // copy gradients to save the analytical solution
        // (otherwise we'll affect them with the code below)
        std::list<real_t> grad;
        for (auto dparam : dparams) {
            std::copy(dparam.begin(), dparam.end(), std::inserter(grad, grad.end()));
        }
        
        // perturb each parameter and evaluate fn()
        auto grad_pos = grad.begin();
        for (auto &param : params) {
            for (std::size_t i = 0; i < param.size(); i++) {
                real_t old_value = param[i];
                param[i] += eps;
                real_t pvalue = fn(input);
                
                param[i] = old_value-eps;
                real_t nvalue = fn(input);
                param[i] = old_value;
                
                real_t numerical_diff = (pvalue - nvalue) / (2*eps);
                real_t analytical_diff = *grad_pos++;
                real_t diff = fabs(numerical_diff - analytical_diff);

                result.push_back(diff);
            }
        }
        
        return std::move(result);
    }
}