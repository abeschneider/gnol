//
//  concat.cpp
//  rnn
//
//  Created by Abe Schneider on 9/16/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#include "concat.hpp"

using namespace gnol;

InputModule::InputModule(size_t size):
    GradientModule(size, size) {}

fmat &InputModule::forward(const fmat &input) {
    output = input;
    return output;
}

fmat &InputModule::backward(const fmat &input, const fmat &grad_output) {
    grad_input = grad_output;
    return grad_input;
}

//     GradientModule(modules.front()->get_input_size(), modules.back()->get_output_size()) {}

std::size_t add_module_sizes(const std::list<std::shared_ptr<GradientModule>> &modules) {
    std::size_t sz = 0;
    for (auto mod : modules)
        sz += mod->get_output_size()[0];
    
    return sz;
}

ConcatModule::ConcatModule(std::list<std::shared_ptr<GradientModule>> modules):
    modules(modules),
    GradientModule(add_module_sizes(modules), add_module_sizes(modules))
{
    output.resize(get_input_size()[0]);
}

fmat &ConcatModule::forward(const fmat &input) {
    fmat::iterator pos = output.begin();
    for (auto mod : modules) {
        mod->forward(input);
        
        // copy result of forward to section of output
        pos = std::copy(mod->get_output().begin(),
                        mod->get_output().end(),
                        pos);        
    }
    
    return output;
}

fmat &ConcatModule::backward(const fmat &input, const fmat &grad_output) {
//    fmat::const_iterator pos = grad_output.begin();
    std::size_t i = 0;
    
    for (auto mod : modules) {
        if (mod->get_input_size().dims() == 1) {
            const std::size_t output_size = mod->get_output_size()[0];
            fvec ginput(mod->get_output_size()[0]);
            grad_input += mod->backward(input, grad_output(span(i, i+output_size-1), 0));
            i += output_size;
        } else {
            // TODO            
        }
    }
    
    return grad_input;
}

parameter_list ConcatModule::flatten_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto mod : modules) {
        auto mod_params = mod->flatten_parameters();
        std::copy(mod_params.begin(), mod_params.end(), insert);
    }
    
    return params;
}

parameter_list ConcatModule::flatten_deriv_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto mod : modules) {
        auto mod_params = mod->flatten_deriv_parameters();
        std::copy(mod_params.begin(), mod_params.end(), insert);
    }
    
    return params;
}

