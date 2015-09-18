//
//  check_gradient.cpp
//  rnn
//
//  Created by Abe Schneider on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "check_gradient.hpp"

//using namespace gnol;
namespace gnol {
    std::list<float> check_gradient(std::function<float (const fvec &)> fn,
                                    GradientModule &mod,
                                    const fvec &input,
                                    float eps)
    {
        std::list<float> result;
        
        parameter_list params = mod.flatten_parameters();
        parameter_list dparams = mod.flatten_deriv_parameters();
        
        // calculate gradients
        fn(input);
        
        // copy gradients (otherwise we'll affect them with the code below)
        std::list<float> grad;
        std::insert_iterator<std::list<float>> grad_insert(grad, grad.end());
        for (auto &param : dparams) {
            std::copy(param.begin(), param.end(), grad_insert);
        }
        
        // perturb each parameter and evaluate fn()
        auto grad_pos = grad.begin();
        for (auto &param : params) {
            for (std::size_t i = 0; i < param.size(); i++) {
                auto old_value = param[i];
                param[i] += eps;
                auto pvalue = fn(input);
                param[i] = old_value;
                
                param[i] -= eps;
                auto nvalue = fn(input);
                param[i] = old_value;
                
                float numerical_diff = (pvalue - nvalue) / (2*eps);
                float calc_diff = *grad_pos++;
                float diff = fabs(numerical_diff - calc_diff);
                result.push_back(diff);
            }
        }
        
        return std::move(result);
    }
}