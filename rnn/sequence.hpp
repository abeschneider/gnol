//
//  sequence.h
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__sequence__
#define __rnn__sequence__

#include "module.hpp"

namespace gnol {
    class SequenceModule: public GradientModule {
    protected:
        std::list<std::shared_ptr<GradientModule>> modules;
    public:
        SequenceModule(std::list<std::shared_ptr<GradientModule>> modules);
        
        void clear();
        fmat &forward(const fmat &input);
        fmat &backward(const fmat &input, const fmat &grad_output);
        parameter_list flatten_parameters();
        parameter_list flatten_deriv_parameters();
    };
}

#endif /* defined(__rnn__sequence__) */
