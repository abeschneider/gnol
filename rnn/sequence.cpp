//
//  sequence.cpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "sequence.hpp"

namespace gnol {
    SequenceModule::SequenceModule(list_t modules):
        modules(modules),
        GradientModule(modules.front()->get_input_size(), modules.back()->get_output_size()) {}


    SequenceModule::SequenceModule(name_list_t modules):
        GradientModule(modules.front().second->get_input_size(),
                       modules.back().second->get_output_size())
    {
        for (auto named_module : modules) {
            this->modules.push_back(named_module.second);
            names[named_module.first] = named_module.second;
        }
    }

    void SequenceModule::clear() {
        grad_input.zeros();
        
        for (auto mod : modules)
            mod->clear();
    }

    matrix_t &SequenceModule::forward(const matrix_t &input) {
        *output = input;
        for (auto mod : modules)
            *output = mod->forward(*output);
        
        return *output;
    }

    matrix_t &SequenceModule::backward(const matrix_t &input, const matrix_t &grad_output) {
        matrix_t ginput = grad_output;
        
        // go until the first module (need to handle that separately
        auto last = modules.rend();
        --last;
        
        for (auto pos = modules.rbegin(); pos != last; pos++) {
            auto prev = pos;
            ++prev;
            
            ginput = (*pos)->backward(*(*prev)->get_output(), ginput);
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

    std::shared_ptr<SequenceModule>
    make_sequence(std::initializer_list<SequenceModule::ptr_t> modules) {
        return std::shared_ptr<SequenceModule>(new SequenceModule(modules));
    }

    std::shared_ptr<SequenceModule>
    make_sequence(std::initializer_list<SequenceModule::pair_t> modules) {
        return std::shared_ptr<SequenceModule>(new SequenceModule(modules));
    }
}