//
//  concat.hpp
//  rnn
//
//  Created by Abe Schneider on 9/16/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#ifndef concat_hpp
#define concat_hpp

#include "module.hpp"

namespace gnol {
    class InputModule: public GradientModule {
    public:
        InputModule(ssize_t<1> size);
        InputModule(ssize_t<2> size);
        matrix_t &forward(const matrix_t &input);
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output);
        parameter_list flatten_parameters() { return empty_parameter_list; }
        parameter_list flatten_deriv_parameters() { return empty_parameter_list; }
    };
    
    /*!
     ConcatModule applies each module to a single input and concatenates 
     the output.
     
     \code
     auto concat = make_concat({
        make_module<LinearModule>(size(10, 5)),
        make_module<LinearModule>(size(10, 5))
     });
     
     vector_t data(10);
     data.urand();
     vector_t output = concat.forward(data);
     
     std::cout << output.n_rows << std::endl;
     // 10
     */
    class ConcatModule: public GradientModule {
        std::list<std::shared_ptr<GradientModule>> modules;
    public:
        ConcatModule(std::list<std::shared_ptr<GradientModule>> modules);
        matrix_t &forward(const matrix_t &input);
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output);
        parameter_list flatten_parameters();
        parameter_list flatten_deriv_parameters();
    };
    
    std::shared_ptr<ConcatModule> make_concat(std::list<std::shared_ptr<GradientModule>> modules) {
        return std::shared_ptr<ConcatModule>(new ConcatModule(modules));
    }
    
    /*!
     JoinModule takes the output of all modules and concatenates them into
     a single vector.
     */
    class JoinModule: public GradientModule {
        std::list<std::shared_ptr<GradientModule>> modules;
    public:
        JoinModule(std::list<std::shared_ptr<GradientModule>> modules);
        matrix_t &forward(const matrix_t &input);
        matrix_t &backward(const matrix_t &input, const matrix_t &grad_output);
        parameter_list flatten_parameters();
        parameter_list flatten_deriv_parameters();
    };
}

#endif /* concat_hpp */
