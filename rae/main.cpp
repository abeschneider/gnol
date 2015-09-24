//
//  main.cpp
//  rae
//
//  Created by Abe Schneider on 9/17/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include <string>
#include <boost/variant.hpp>

#include "module.hpp"
#include "criterion.hpp"
#include "check_gradient.hpp"
#include "linear.hpp"
#include "sequence.hpp"
#include "concat.hpp"
#include "reshape.hpp"
#include "activation.hpp"

using namespace gnol;

std::shared_ptr<SequenceModule>
make_autoencoder(variable<matrix_t> &weight,
                 variable<matrix_t> &grad_weight)
{
    auto input = make_module<InputModule>(size(weight->n_rows));
    auto encoder = make_module<LinearModule>(share(weight),
                                             share(grad_weight));
    
    auto decoder = make_module<TransposedLinearModule>(share(weight),
                                                       share(grad_weight));

    auto encoder_sigmoid =
        make_module<SigmoidModule>(encoder->get_output_size());
    
    auto decoder_sigmoid =
        make_module<SigmoidModule>(decoder->get_output_size());
    
    // tie the error to the input so we're solving for the
    // reconstruction error
    auto error =
        std::make_shared<CriterionModule<L2Op, L2Gradient>>(
            share(input->get_output()));

    auto autoencoder = make_sequence({
        {"input", input},
        {"encoder_activation", encoder},
        {"encoder", encoder_sigmoid},
        {"decoder_activation", decoder},
        {"decoder", decoder},
        {"error", error}
    });
    
    // TODO: need a way to attach a criterion to this

    return std::move(autoencoder);
}

int main(int argc, const char * argv[]) {
    // S = ((a b) (c d))
    
    // weight matrix shared with all autoencoder modules
    variable<matrix_t> weight(size(10, 5));
    variable<matrix_t> grad_weight(size(10, 5));
    
    // (a b)
    auto ae1 = make_autoencoder(weight, grad_weight);
    auto ae2 = make_autoencoder(weight, grad_weight);
    auto concat1 = make_concat({(*ae1)["encoder"], (*ae2)["encoder"]});
    
    // (c d)
    auto ae3 = make_autoencoder(weight, grad_weight);
    auto ae4 = make_autoencoder(weight, grad_weight);
    auto concat2 = make_concat({(*ae3)["encoder"], (*ae4)["encoder"]});

    // main network
    auto concat3 = make_concat({concat1, concat2});

    vector_t a(10), b(10), c(10), d(10);
    vector_t input = concat({a, b, c, d});
    auto output = concat3->forward(input);
    
    std::cout << output << std::endl;
    
    return 0;
}
