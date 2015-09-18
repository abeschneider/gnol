//
//  tests.cpp
//  rnn
//
//  Created by Abraham Schneider on 8/26/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include <armadillo>

#include "rnn.hpp"

using namespace gnol;

float distance(const fmat &a, const fmat &b) {
    return as_scalar(sum(sum(sqrt(pow(a - b, 2)))));
}

bool is_close(const fmat &a, const fmat &b, float eps=10e-4) {
    return distance(a, b) < eps;
}

void test_gradient2(GradientModule &mod, float eps=10e-4) {
    L2Loss loss;
    
    fvec input;
    
    input.set_size(mod.get_input_size()[0]);
    input.randu();
    
    fvec target(mod.get_output_size()[0]);
    target.randu();
    
    auto eval_fn = [&mod, &loss, target](const fvec &x) -> float {
        // make sure we haven't accumlated any error yet
        mod.clear();
        
        auto result = mod.forward(x);
        auto error = loss.forward(result, target);
        auto grad = loss.backward(result, target);
        mod.backward(x, grad);
        
        return error;
    };
    
    auto result = check_gradient(eval_fn, mod, input, eps);
    ASSERT_GT(result.size(), 0);
    
    for (float val : result) {
        ASSERT_LT(val, eps);
    }
}

TEST(variable, vector) {
    variable<fvec> var(5);
    var->ones();
    
    fvec expected = {1.0, 1.0, 1.0, 1.0, 1.0};
    ASSERT_TRUE(is_close(*var, expected));
}

TEST(variable, matrix) {
    variable<fmat> var({5, 5});
    var->ones();
    
    fmat expected(5, 5);
    expected.fill(1.0);
    ASSERT_TRUE(is_close(*var, expected));
    
    variable<fmat> var2(var);
}

TEST(variable, shared_matrix) {
    variable<fmat> var1({5, 5});
    
    // tell var2 to share the storage of var1
    variable<fmat> var2(share(var1));

    // if we change var1, var2 should change
    var1->ones();
    ASSERT_TRUE(is_close(*var2, *var1));
    
    // and vice versa
    var2->zeros();
    ASSERT_TRUE(is_close(*var2, *var1));
}

struct test_struct {
    variable<fmat> var;

    
    test_struct(variable<fmat> &v):
        var(std::move(v)) {}

    test_struct(variable<fmat> &&v):
        var(std::move(v)) {}
};

// test that if we use the move semantics the shared memory will be kept
TEST(variable, shared_matrix2) {
    variable<fmat> var({5, 5});
    
    // test with l-value reference
    auto shared_var = share(var);
    test_struct ts(shared_var);
    
    var->zeros();
    ASSERT_TRUE(is_close(*var, *ts.var));
    
    // test with move semantics
    test_struct ts2(share(var));
    
    var->ones();
    ASSERT_TRUE(is_close(*var, *ts2.var));
}

TEST(LinearModule, Initialize) {
    LinearModule linear({3, 5});
}

TEST(LinearModule, forward) {
    LinearModule linear({3, 5});

    linear.get_params().weight->eye();
    linear.get_params().bias->zeros();
    
    fvec input = {0.1, 0.2, 0.3};
    auto output = linear.forward(input);
    
    fvec expected = {0.1, 0.2, 0.3, 0, 0};
    ASSERT_TRUE(is_close(output, expected));
}

TEST(LinearModule, Backward) {
    LinearModule linear({3, 5});
    linear.get_params().weight->eye();
    *linear.get_params().weight *= 0.1;
    linear.get_params().bias->zeros();
    
    linear.clear();
    
    fvec input = {0.1, 0.2, 0.3};
    fvec grad_output = {1.0, 1.0, 1.0, 1.0, 1.0};
    auto output = linear.forward(input);
    
    auto grad_input = linear.backward(input, grad_output);
    
    fvec expected = {0.1, 0.1, 0.1};
    ASSERT_TRUE(is_close(grad_input, expected));
}

TEST(LinearModule, GradCheck) {
    LinearModule linear({3, 5});
    test_gradient2(linear);
}

TEST(SequenceModule, Initialize) {
    auto linear =
        make_module<LinearModule>(size(3, 5));
    auto sigmoid = make_module<SigmoidModule>(5);
    
    SequenceModule seq({linear, sigmoid});
}

TEST(SequenceModule, GradCheck) {
    auto linear =
    make_module<LinearModule>(size(3, 5));
    auto sigmoid = make_module<SigmoidModule>(5);
    
    SequenceModule seq({linear, sigmoid});
    test_gradient2(seq);
}

TEST(LinearParams, Initialize) {
    LinearParams params({3, 5});
}

TEST(LinearParams, SharedWeights) {
    LinearParams params({3, 5});
    LinearParams shared_params(share(params.weight),
                               make_vector(params.weight->n_rows));
    params.weight->ones();

    // make sure shared_parameters match params
    ASSERT_TRUE(is_close(*shared_params.weight, *params.weight));
}

TEST(LinearModule, SharedWeights) {
    // create an auto-encoder with tied weights
    LinearModule encoder({3, 5});
    
    // shared weight matrix
    LinearParams params(share(encoder.get_params().weight),
                        make_vector(encoder.get_params().weight->n_rows));
    
    // shared gradient weight matrix
    LinearGradParams gparam(share(encoder.get_grad_params().weight),
                            make_vector(encoder.get_grad_params().weight->n_rows));
    
    // all ops will be transposed
    TransposedLinearModule decoder(std::move(params), std::move(gparam));

    encoder.get_params().weight->eye();
    encoder.get_params().bias->zeros();

    // bias of decoder should be the size of the decoder's output
    ASSERT_EQ(decoder.get_params().bias->n_rows, decoder.get_output_size()[0]);

    // test forward direction
    fvec input = {0.1, 0.2, 0.3};
    auto hidden = encoder.forward(input);
    auto output = decoder.forward(hidden);
    
    fvec expected_hidden = {0.1, 0.2, 0.3, 0, 0};
    fvec expected_output = {0.1, 0.2, 0.3};
        
    ASSERT_TRUE(is_close(hidden, expected_hidden));
    ASSERT_TRUE(is_close(output, expected_output));
    
    // test backward direction
    fvec grad_output = {0.0, 0.0, 0.0};
    auto grad_input1 = decoder.backward(hidden, grad_output);
    auto grad_input2 = encoder.backward(input, grad_input1);
}

TEST(LinearModule, SharedWeightsGradCheck) {
    // create an auto-encoder with tied weights
    auto encoder = make_module<LinearModule>(size(3, 5));
    
    // shared weight matrix
    LinearParams params(share(encoder->get_params().weight),
                        make_vector(encoder->get_params().weight->n_rows));
    
    // shared gradient weight matrix
    LinearGradParams gparam(share(encoder->get_grad_params().weight),
                            make_vector(encoder->get_grad_params().weight->n_rows));
    
    // all ops will be transposed
    auto decoder =
        std::make_shared<TransposedLinearModule>(std::move(params),
                                                 std::move(gparam));
    
    auto encoder_sigmoid = make_module<SigmoidModule>(5);
    auto decoder_sigmoid = make_module<SigmoidModule>(3);
    SequenceModule seq({encoder, encoder_sigmoid, decoder, decoder_sigmoid});
    
    test_gradient2(seq);
}

TEST(ReshapeModule, Initialize) {
    ReshapeModule<2> reshape(size(2, 2), size(4, 1));
}

TEST(ReshapeModule, Forward) {
    auto reshape = make_module<ReshapeModule<2>>(size(3, 3), size(9, 1));
    auto linear = make_module<LinearModule>(size(9, 3));
    auto sigmoid = make_module<SigmoidModule>(3);
    
    linear->get_params().weight->ones();
    linear->get_params().bias->zeros();
    
    SequenceModule seq({reshape, linear, sigmoid});
    
    // 2D data gets reshaped to 1D for linear layer
    fmat data = {   {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}};
    
    seq.forward(data);
}

TEST(ReshapeModule, GradCheck) {
    auto reshape = make_module<ReshapeModule<2>>(size(3, 3), size(9, 1));
    auto linear = make_module<LinearModule>(size(9, 3));
    auto sigmoid = make_module<SigmoidModule>(3);
    
    SequenceModule seq({reshape, linear, sigmoid});
    test_gradient2(seq);
}

TEST(ConcatenateModule, Forward) {
    auto linear1 = make_module<LinearModule>(size(10, 5));
    fmat eye1 = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0 ,0, 0, 1},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    
    fmat eye2 = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0 ,0, 0, 1},
    };

    *linear1->get_params().weight = eye1;
    linear1->get_params().bias->zeros();
    
    auto linear2 = make_module<LinearModule>(size(10, 5));
    *linear2->get_params().weight = eye2;
    linear2->get_params().bias->zeros();
    
    ConcatModule concat({linear1, linear2});

    fvec input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto output = concat.forward(input);
    
    ASSERT_TRUE(is_close(output, input));
}

TEST(ConcatModule, Backward) {
    auto linear1 = make_module<LinearModule>(size(10, 5));
    linear1->get_params().weight->eye();
    linear1->get_params().bias->zeros();

    auto linear2 = make_module<LinearModule>(size(10, 5));
    linear2->get_params().weight->eye();
    linear2->get_params().bias->zeros();

    ConcatModule concat({linear1, linear2});
    
    fvec input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto output = concat.forward(input);
    
    fvec grad_output(10); // = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    grad_output.ones();
    auto grad_input = concat.backward(input, grad_output);
    
    fvec expected = {2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ASSERT_TRUE(is_close(grad_input, expected));
}

TEST(ConcatModule, GradCheck) {
    auto linear1 = make_module<LinearModule>(size(10, 5));
    linear1->get_params().weight->eye();
    linear1->get_params().bias->zeros();
    
    auto linear2 = make_module<LinearModule>(size(10, 5));
    linear2->get_params().weight->eye();
    linear2->get_params().bias->zeros();
    
    ConcatModule concat({linear1, linear2});

    test_gradient2(concat);
}


