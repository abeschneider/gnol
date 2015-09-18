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
        InputModule(size_t size);
        fmat &forward(const fmat &input);
        fmat &backward(const fmat &input, const fmat &grad_output);
        parameter_list flatten_parameters() { return empty_parameter_list; }
        parameter_list flatten_deriv_parameters() { return empty_parameter_list; }
    };
    
    class ConcatModule: public GradientModule {
        std::list<std::shared_ptr<GradientModule>> modules;
    public:
        ConcatModule(std::list<std::shared_ptr<GradientModule>> modules);
        fmat &forward(const fmat &input);
        fmat &backward(const fmat &input, const fmat &grad_output);
        parameter_list flatten_parameters(); //{ return empty_parameter_list; }
        parameter_list flatten_deriv_parameters(); //{ return empty_parameter_list; }
    };
}

#endif /* concat_hpp */
