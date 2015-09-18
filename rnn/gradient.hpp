//
//  grad.h
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__gradient__
#define __rnn__gradient__

#include "op.hpp"

#pragma GCC visibility push(default)

namespace weight {
//    template <typename OpT>
//    class Gradient {
//    protected:
//        OpT &op;
//    public:
//        Gradient(OpT &op): op(op) {}
//        
//        virtual void reset() = 0;
//        virtual void operator ()(const fmat &input,
//                                 const fmat &output,
//                                 const fmat &grad_output,
//                                 fmat &grad_input) = 0;
//        virtual parameter_list get_deriv_parameters() = 0;
//    };
    
    class LinearGradient {//{: public Gradient<LinearOp> {
    private:
        LinearOp &op;
        fmat grad_weight;
        fvec grad_bias;
    public:
        LinearGradient(LinearOp &op);
        
        void reset();
        
        void operator ()(const fvec &input,
                         const fvec &output,
                         const fvec &grad_output,
                         fvec &grad_input);
        
        void operator ()(const fmat &input,
                         const fmat &output,
                         const fmat &grad_output,
                         fmat &grad_input);
        
        parameter_list get_deriv_parameters();
    };
    
    class TransposeGradient {
    private:
        TransposeOp &op;
        fmat grad_weight;
        fvec grad_bias;
    public:
        TransposeGradient(TransposeOp &op);
        
        void reset();
        
        void operator ()(const fvec &input,
                         const fvec &output,
                         const fvec &grad_output,
                         fvec &grad_input);
        
        void operator ()(const fmat &input,
                         const fmat &output,
                         const fmat &grad_output,
                         fmat &grad_input);
        
        parameter_list get_deriv_parameters();
    };

    class SigmoidGradient { //: public Gradient<SigmoidOp> {
    private:
        SigmoidOp op;
    public:
        SigmoidGradient(SigmoidOp &op);
        void reset() {}
        void operator ()(const fmat &input,
                         const fmat &output,
                         const fmat &grad_output,
                         fmat &grad_input);
        
        parameter_list get_deriv_parameters() { return parameter_list(); }
    };
}

#pragma GCC visibility pop

#endif /* defined(__rnn__grad__) */
