//
//  sequence.h
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__sequence__
#define __rnn__sequence__

#include <map>

#include "module.hpp"

namespace gnol {
    class SequenceModule: public GradientModule {
    public:
        typedef std::shared_ptr<GradientModule> ptr_t;
        typedef std::pair<std::string, ptr_t> pair_t;
        typedef std::vector<ptr_t> list_t;
        typedef std::vector<pair_t> name_list_t;
    protected:
        list_t modules;
        std::map<std::string, ptr_t> names;
    public:
        SequenceModule(list_t modules);
        SequenceModule(name_list_t modules);
        
        ptr_t operator [](const std::string &name) { return names[name]; }
        ptr_t operator [](std::size_t index) { return modules[index]; }
        
        void clear();
        matrix_t &forward(const matrix_t &input);
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output);
        parameter_list flatten_parameters();
        parameter_list flatten_deriv_parameters();
    };
    
    std::shared_ptr<SequenceModule>
    make_sequence(std::initializer_list<SequenceModule::ptr_t> modules);
    
    std::shared_ptr<SequenceModule>
    make_sequence(std::initializer_list<SequenceModule::pair_t> modules);
}

#endif /* defined(__rnn__sequence__) */
