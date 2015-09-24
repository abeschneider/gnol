//
//  concat.cpp
//  rnn
//
//  Created by Abe Schneider on 9/16/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#include "concat.hpp"

using namespace gnol;

InputModule::InputModule(ssize_t<1> size):
    GradientModule(size, size) {}

InputModule::InputModule(ssize_t<2> size):
    GradientModule(size, size) {}

matrix_t &InputModule::forward(const matrix_t &input) {
    *output = input;
    return *output;
}

matrix_t &InputModule::backward(const matrix_t &input, const matrix_t &grad_output) {
    grad_input = grad_output;
    return grad_input;
}

std::size_t add_module_input_sizes(const std::list<std::shared_ptr<GradientModule>> &modules) {
    std::size_t sz = 0;
    for (auto mod : modules)
        sz += mod->get_input_size()[0];
    
    return sz;
}

std::size_t add_module_output_sizes(const std::list<std::shared_ptr<GradientModule>> &modules) {
    std::size_t sz = 0;
    for (auto mod : modules)
        sz += mod->get_output_size()[0];
    
    return sz;
}

ConcatModule::ConcatModule(std::list<std::shared_ptr<GradientModule>> modules):
    modules(modules),
    GradientModule(modules.front()->get_input_size(),
                   add_module_output_sizes(modules)) {}

matrix_t &ConcatModule::forward(const matrix_t &input) {
    matrix_t::iterator pos = output->begin();
    for (auto mod : modules) {
        mod->forward(input);
        
        // copy result of forward to section of output
        pos = std::copy(mod->get_output()->begin(),
                        mod->get_output()->end(),
                        pos);        
    }
    
    return *output;
}

matrix_t &ConcatModule::backward(const matrix_t &input, const matrix_t &grad_output) {
    std::size_t i = 0;
    
    for (auto mod : modules) {
        if (mod->get_input_size().dims() == 1) {
            const std::size_t output_size = mod->get_output_size()[0];
            vector_t ginput(mod->get_output_size()[0]);
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

JoinModule::JoinModule(std::list<std::shared_ptr<GradientModule>> modules):
    modules(modules),
    GradientModule(add_module_input_sizes(modules),
                   add_module_output_sizes(modules)) {}

matrix_t &JoinModule::forward(const matrix_t &input) {
    std::size_t i = 0;
    std::size_t j = 0;
    
    for (auto mod : modules) {
        const std::size_t input_size = mod->get_input_size()[0];
        const std::size_t output_size = mod->get_output_size()[0];
        
        // forward slice of input to each module
        mod->forward(input(span(i, i+input_size-1), 0));
        
        // copy result of forward to slice of output
        (*output)(span(j, j+output_size-1), 0) = *mod->get_output();
        
        // increment index into input
        i += input_size;
        j += output_size;
    }
    
    return *output;
}

matrix_t &JoinModule::backward(const matrix_t &input, const matrix_t &grad_output) {
    std::size_t i = 0;
    std::size_t j = 0;
    
    for (auto mod : modules) {
        if (mod->get_input_size().dims() == 1) {
            const std::size_t input_size = mod->get_input_size()[0];
            const std::size_t output_size = mod->get_output_size()[0];

            // backward slice of grad_output to each module with the same
            // input slice that was given in the forward phase
            mod->backward(input(span(i, i+input_size-1), 0),
                          grad_output(span(j, j+output_size-1), 0));
            
            // copy result of backward to slice of grad_input
            grad_input(span(i, i+input_size-1), 0) = mod->get_grad_input();
            
            // increment index to input and grad_output
            i += input_size;
            j += output_size;
        } else {
            // TODO
        }
    }
    
    return grad_input;
}

parameter_list JoinModule::flatten_parameters() {
//    parameter_list params;
//    std::insert_iterator<parameter_list> insert(params, params.end());
//    
//    for (auto mod : modules) {
//        auto mod_params = mod->flatten_parameters();
//        std::copy(mod_params.begin(), mod_params.end(), insert);
//    }
//    
//    return params;
    return modules.front()->flatten_parameters();
}

parameter_list JoinModule::flatten_deriv_parameters() {
//    parameter_list params;
//    std::insert_iterator<parameter_list> insert(params, params.end());
//    
//    for (auto mod : modules) {
//        auto mod_params = mod->flatten_deriv_parameters();
//        std::copy(mod_params.begin(), mod_params.end(), insert);
//    }
//    
//    return params;
     return modules.front()->flatten_deriv_parameters();
}
