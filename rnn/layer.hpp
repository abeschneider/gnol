//
//  layer.h
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__layer__
#define __rnn__layer__

#include "op.hpp"
#include "gradient.hpp"

#include <memory>

#pragma GCC visibility push(default)

namespace weight {
    class Layer {
    public:
        typedef std::function<const fmat &(const fmat &)> forward_t;
        typedef std::function<const fmat &(const fmat &, const fmat &)> backward_t;
    private:
        forward_t forward_fn;
        backward_t backward_fn;
    protected:
        fmat output;
        fmat grad_input;
        size_t input_size;
        size_t output_size;
    protected:
        void initialize() {
            if (input_size.dims() == 1) {
                grad_input.resize(input_size[0]);
            } else {
                grad_input.resize(input_size[0], input_size[1]);
            }
            
            if (output_size.dims() == 1) {
                output.resize(output_size[0]);
            } else {
                output.resize(output_size[0], output_size[1]);
            }
            
            output.zeros();
            grad_input.zeros();
        }
        
        void initialize(size_t input_size, size_t output_size) {
            this->input_size = input_size;
            this->output_size = output_size;
            initialize();
        }
    public:
        Layer(forward_t forward_fn, backward_t backward_fn):
            forward_fn(forward_fn),
            backward_fn(backward_fn) {}
        
        Layer(forward_t forward_fn,
              backward_t backward_fn,
              size_t input_size,
              size_t output_size):
            forward_fn(forward_fn),
            backward_fn(backward_fn),
            input_size(input_size),
            output_size(output_size)
        {
            initialize();
        }

        const fmat &forward(const fmat &input) {
            return forward_fn(input);
        }
        
        const fmat &backward(const fmat &input, const fmat &grad_output) {
            return backward_fn(input, grad_output);
        }
//        virtual const fmat &forward(const fmat &input) = 0;
//        virtual const fmat &backward(const fmat &input,
//                                     const fmat &grad_output) = 0;
        
        virtual parameter_list get_parameters() = 0;
        virtual parameter_list get_deriv_parameters() = 0;

        const fmat &get_output() { return output; }
        const fmat &get_grad_input() { return grad_input; }
        
        size_t get_input_size() { return input_size; }
        size_t get_output_size() { return output_size; }
        
        void reset() { grad_input.zeros(); }
    };


    template <typename OpT, typename GradT>
    class OpLayer: public Layer {
    protected:
        std::shared_ptr<OpT> op;
        std::shared_ptr<GradT> grad;
    protected:
        template <typename... Args>
        void initialize(Args... args) {
            op = std::make_shared<OpT>(args...);
            grad = std::make_shared<GradT>(*op);
        }
    public:
        OpLayer():
            Layer(
                  [this](const fmat &input) -> const fmat & {
                      return this->forward(input);
                  },
                  [this](const fmat &input, const fmat &grad_output) -> const fmat & {
                      return this->backward(input, grad_output);
                  }) {}

        OpLayer(size_t input_size, size_t output_size):
            Layer(
              [this](const fmat &input) -> const fmat & {
                  return this->forward(input);
              },
              [this](const fmat &input, const fmat &grad_output) -> const fmat & {
                  return this->backward(input, grad_output);
              },
              input_size,
              output_size) {}
        
        OpLayer(std::shared_ptr<OpT> op, std::shared_ptr<GradT> grad):
            op(op),
            grad(grad),
            Layer(
              [this](const fmat &input) -> const fmat & {
                  return this->forward(input);
              },
              [this](const fmat &input, const fmat &grad_output) -> const fmat & {
                  return this->backward(input, grad_output);
              },
              op->input_size(),
              op->output_size()) {}
        
        const fmat &forward(const fmat &input);
        const fmat &backward(const fmat &input, const fmat &grad_output);
        
        std::shared_ptr<OpT> get_op() { return op; }
        std::shared_ptr<GradT> get_grad() { return grad; }
        
        virtual parameter_list get_parameters() {
            return op->get_parameters();
        }
        
        virtual parameter_list get_deriv_parameters() {
            return grad->get_deriv_parameters();
        }
    };
    
    template <typename OpT, typename GradT>
    const fmat &OpLayer<OpT, GradT>::forward(const fmat &input) {
        (*op)(input, output);
        return output;
    }
    
    template <typename OpT, typename GradT>
    const fmat &OpLayer<OpT, GradT>::backward(const fmat &input,
                                              const fmat &grad_output)
    {
        (*grad)(input, output, grad_output, grad_input);
        return grad_input;
    }

    
    // create a new layer that shared the parameters of the layer
    // passed as the parameter
    template <typename LayerT>
    std::shared_ptr<LayerT> share(std::shared_ptr<LayerT> layer) {
        return std::make_shared<LayerT>(layer->get_op(), layer->get_grad());
    }

    class LinearLayer: public OpLayer<LinearOp, LinearGradient> {
    public:
        LinearLayer(ssize_t<1> input_size, ssize_t<1> output_size);
        LinearLayer(std::shared_ptr<LinearOp> op, std::shared_ptr<LinearGradient> grad);
        
        
        fmat &forward(const fmat &input) {
            (*op)(input, output);
            return output;
        }
        
        fmat &backward(const fmat &input, const fmat &grad_output) {
            (*grad)(input, output, grad_output, grad_input);
            return grad_input;
        }

        
        fmat &weight() { return op->weight; }
        fmat &bias() { return op->bias; }
    };
    
    class SigmoidLayer: public OpLayer<SigmoidOp, SigmoidGradient> {
    public:
        SigmoidLayer(ssize_t<1> size);
    };

    class SequenceLayer: public Layer {
    protected:
        std::list<std::shared_ptr<Layer>> layers;
    public:
        SequenceLayer(std::initializer_list<std::shared_ptr<Layer>> layers);
        SequenceLayer(std::list<std::shared_ptr<Layer>> layers);
        
        const fmat &forward(const fmat &input);
        const fmat &backward(const fmat &input, const fmat &grad_output);
        
        parameter_list get_parameters();
        parameter_list get_deriv_parameters();
    };
    
    template <std::size_t D>
    class ReshapeLayer: public Layer {};
    
    template <>
    class ReshapeLayer<2>: public Layer {
    protected:
        fmat reshaped_input;
        fmat reshaped_grad_input;
    public:
        ReshapeLayer(ssize_t<2> input_size, ssize_t<2> output_size):
            Layer(
              [this](const fmat &input) -> const fmat & {
                  return this->forward(input);
              },
              [this](const fmat &input, const fmat &grad_output) -> const fmat & {
                  return this->backward(input, grad_output);
              },
              input_size,
              output_size) {}
        
        const fmat &forward(const fmat &input) {
            reshaped_input = input.submat(0, 0, input.n_rows-1, input.n_cols-1);
            reshaped_input.reshape(output_size[0], output_size[1]);
            return reshaped_input;
        }

        const fmat &backward(const fmat &input, const fmat &grad_output) {
            reshaped_grad_input = grad_output.submat(0, 0, grad_output.n_rows - 1, grad_output.n_cols - 1);
            reshaped_grad_input.reshape(input_size[0], input_size[1]);
            return reshaped_grad_input;
        }
        
        parameter_list get_parameters() {
            return parameter_list();
        }
        
        parameter_list get_deriv_parameters() {
            return parameter_list();
        }
    };
    
    class TransposeLayer: public OpLayer<TransposeOp, TransposeGradient> {
    public:
        TransposeLayer(std::shared_ptr<LinearLayer> layer);
    };
    
//    class ConcatLayer: public Layer {
//    protected:
//        std::list<std::shared_ptr<Layer>> layers;
//    public:
//        ConcatLayer(std::initializer_list<std::shared_ptr<Layer>> layers);
//        ConcatLayer(std::list<std::shared_ptr<Layer>> layers);
//        
//        const fmat &forward(const fmat &input);
//        const fmat &backward(const fmat &input, const fmat &grad_output);
//        
//        parameter_list get_parameters();
//        parameter_list get_deriv_parameters();
//    };
}

#pragma GCC visibility pop

#endif /* defined(__rnn__layer__) */
