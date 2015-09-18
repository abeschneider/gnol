//
//  criterion.h
//  rnn
//
//  Created by Abe Schneider on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__criterion__
#define __rnn__criterion__

#include <armadillo> 

using namespace arma;

namespace gnol {
    class Criterion {
    public:
        virtual float forward(const fvec &input, const fvec &target) = 0;
        virtual fvec backward(const fvec &input, const fvec &target) = 0;
    };
    
    class L2Loss: public Criterion {
    private:
        fvec grad_input;
    public:
        float forward(const fvec &input, const fvec &target) {
            auto result = sum(pow(input - target, 2));
            return std::move(result);
        }
        
        fvec backward(const fvec &input, const fvec &target) {
            grad_input = 2*(input - target);
            return grad_input;
        }
    };
}

#endif /* defined(__rnn__criterion__) */
