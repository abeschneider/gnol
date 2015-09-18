//
//  layer.cpp
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "layer.hpp"

using namespace weight;

/*LinearLayer::LinearLayer(std::shared_ptr<LinearLayer> layer) {
    initialize(layer);
}*/

LinearLayer::LinearLayer(ssize_t<1> input_size, ssize_t<1> output_size):
    OpLayer(input_size, output_size)
{
    initialize(input_size, output_size);
}

LinearLayer::LinearLayer(std::shared_ptr<LinearOp> op, std::shared_ptr<LinearGradient> grad):
    OpLayer(op, grad)
{
    initialize(op->input_size(), op->output_size());
}


/*SigmoidLayer::SigmoidLayer(std::shared_ptr<SigmoidLayer> layer) {
    initialize(layer);
}*/

SigmoidLayer::SigmoidLayer(ssize_t<1> size):
    OpLayer(size, size)
{
    initialize(size);
}

SequenceLayer::SequenceLayer(std::initializer_list<std::shared_ptr<Layer>> layers):
    layers(layers),
    Layer(
      [this](const fmat &input) -> const fmat & {
          return this->forward(input);
      },
      [this](const fmat &input, const fmat &grad_output) -> const fmat & {
          return this->backward(input, grad_output);
      },
      ssize_t<1>(0),
      ssize_t<1>(0))
{
    initialize(this->layers.front()->get_input_size(), this->layers.back()->get_output_size());
}

SequenceLayer::SequenceLayer(std::list<std::shared_ptr<Layer>> layers):
    layers(layers),
    Layer(
      [this](const fmat &input) -> const fmat & {
          return this->forward(input);
      },
      [this](const fmat &input, const fmat &grad_output) -> const fmat & {
          return this->backward(input, grad_output);
      },
      layers.front()->get_input_size(),
      layers.back()->get_output_size()) {}

const fmat &SequenceLayer::forward(const fmat &input) {
    auto layer = layers.begin();
    auto last_layer = layers.end();
    --last_layer;
    
    (*layer)->forward(input);
    for (; layer != last_layer; layer++) {
        auto next_layer = layer;
        ++next_layer;
        (*next_layer)->forward((*layer)->get_output());
    }
    
    output = layers.back()->get_output();
    return output;
}

const fmat &SequenceLayer::backward(const fmat &input, const fmat &grad_output) {
    fmat ginput = grad_output;
    
    // go until the first module (need to handle that separately
    auto last = layers.rend();
    --last;
    
    for (auto pos = layers.rbegin(); pos != last; pos++) {
        auto prev = pos;
        ++prev;
        
        ginput = (*pos)->backward((*prev)->get_output(), ginput);
    }
    ginput = layers.front()->backward(input, ginput);
    
    grad_input += ginput;
    return grad_input;
}

parameter_list SequenceLayer::get_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto layer : layers) {
        auto layer_params = layer->get_parameters();
        std::copy(layer_params.begin(), layer_params.end(), insert);
    }
    
    return params;
}

parameter_list SequenceLayer::get_deriv_parameters() {
    parameter_list params;
    std::insert_iterator<parameter_list> insert(params, params.end());
    
    for (auto layer : layers) {
        auto layer_params = layer->get_deriv_parameters();
        std::copy(layer_params.begin(), layer_params.end(), insert);
    }
    
    return params;
}

TransposeLayer::TransposeLayer(std::shared_ptr<LinearLayer> layer)
{
    op = std::make_shared<TransposeOp>(layer->get_op());
    grad = std::make_shared<TransposeGradient>(*op);
    
    Layer::initialize(op->input_size(), op->output_size());

//    initialize(op->input_size(), op->output_size());
}
