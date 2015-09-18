//
//  sequence.cpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "sequence.hpp"

using namespace gnol;

SequenceModule::SequenceModule(std::list<std::shared_ptr<GradientModule>> modules):
    modules(modules),
    GradientModule(modules.front()->get_input_size(), modules.back()->get_output_size()) {}


void SequenceModule::clear() {
    grad_input.zeros();
    
    for (auto mod : modules)
        mod->clear();
}

fmat &SequenceModule::forward(const fmat &input) {
    output = input;
    for (auto mod : modules)
        output = mod->forward(output);
    
    return output;
}

fmat &SequenceModule::backward(const fmat &input, const fmat &grad_output) {
    fmat ginput = grad_output;
    
    // go until the first module (need to handle that separately
    auto last = modules.rend();
    --last;
    
    for (auto pos = modules.rbegin(); pos != last; pos++) {
        auto prev = pos;
        ++prev;
        
        ginput = (*pos)->backward((*prev)->get_output(), ginput);
    }
    ginput = modules.front()->backward(input, ginput);
    
    grad_input += ginput;
    return grad_input;
}

parameter_list SequenceModule::flatten_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto mod : modules) {
        auto mod_params = mod->flatten_parameters();
        std::copy(mod_params.begin(), mod_params.end(), insert);
    }
    
    return params;
}

parameter_list SequenceModule::flatten_deriv_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto mod : modules) {
        auto mod_params = mod->flatten_deriv_parameters();
        std::copy(mod_params.begin(), mod_params.end(), insert);
    }
    
    return params;
}

