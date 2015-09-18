//
//  op.hpp
//  rnn
//
//  Created by Lais Washington on 8/30/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef rnn_op_hpp
#define rnn_op_hpp

#include <armadillo>
#include <vector>
#include <list>
#include <memory>
#include <functional>
#include <utility>
#include <initializer_list>
#include <array>

#include <boost/range.hpp>

using namespace arma;

#include <memory>

#pragma GCC visibility push(default)

namespace weight {
    typedef std::list<boost::iterator_range<fmat::iterator>> parameter_list;
//    typedef std::array<std::size_t, 2> size_t;
    
    
    class size_t;
    
    template <std::size_t D>
    struct ssize_t {
        friend size_t;
    public:
        typedef std::array<std::size_t, D> list_t;
        typedef typename list_t::iterator iterator;
    private:
        list_t extent;
    public:
        ssize_t() {
            std::fill(extent.begin(), extent.end(), 0);
        }
        
        ssize_t(std::initializer_list<std::size_t> values) {
            std::copy(values.begin(), values.end(), extent.begin());
        }
        
        std::size_t operator[](std::size_t index) { return extent[index]; }
        
        iterator begin() { return extent.begin(); }
        iterator end() { return extent.end(); }
    };
    
    // convenience for 1 dimensional sizes
    template <>
    struct ssize_t<1> {
        friend size_t;
    private:
        std::size_t extent;
    public:
        ssize_t() { extent = 0; }
        
        ssize_t(std::size_t value) {
            extent = value;
        }
        
        // here just for compatability
        std::size_t operator [](std::size_t index) { return extent; }
        
        operator std::size_t() {
            return extent;
        }
        
        std::size_t *begin() { return &extent; }
        std::size_t *end() { return (&extent)+1; }
    };

    
    struct size_t {
    public:
        typedef std::vector<std::size_t> list_t;
        typedef list_t::iterator iterator;
    private:
        std::vector<std::size_t> extent;
    public:
        size_t() {}
        
        size_t(std::initializer_list<std::size_t> values) {
            extent.resize(values.size());
            copy(values.begin(), values.end(), extent.begin());
        }
        
        template <std::size_t D>
        size_t(ssize_t<D> size) {
            extent.resize(D);
            std::copy(size.begin(), size.end(), extent.begin());
        }
        
        size_t &operator =(std::initializer_list<std::size_t> values) {
            extent.resize(values.size());
            copy(values.begin(), values.end(), extent.begin());
            return *this;
        }
        
        size_t &operator =(size_t &size) {
            extent.resize(size.dims());
            copy(size.begin(), size.end(), extent.begin());
            return *this;
        }
        
        std::size_t dims() const { return extent.size(); }
        std::size_t size() const { return extent.size(); }
        
        std::size_t operator [](std::size_t index) { return extent[index]; }

        iterator begin() { return extent.begin(); }
        iterator end() { return extent.end(); }
        /*template <>
        size_t<1>(ssize_t<1> size) {
            extent.resize(1);
            extent[0] = size.extent;
        }*/
    };
    
    class Data {
        
    };
    
    class LinearOp {
    public:
        fmat weight;
        fvec bias;
    public:
        LinearOp(ssize_t<1> input_size, ssize_t<1> output_size);
        
        void operator ()(const fvec &input, fvec &output) {
            output = weight.t()*input + bias;
        }
        
        void operator ()(const fmat &input, fmat &output) {
            output = weight.t()*input + bias;
        }
        
        ssize_t<1> input_size() { return weight.n_rows; }
        ssize_t<1> output_size() { return weight.n_cols; }
        parameter_list get_parameters();
    };
    
    class TransposeGradient;
    class TransposeOp {
        friend TransposeGradient;
    private:
        std::shared_ptr<LinearOp> linear;
        
        // duplicate .. separate out bias layer?
        fvec bias;
    public:
        TransposeOp(std::shared_ptr<LinearOp> op):
            linear(op)
        {
            bias.resize(output_size());
            bias.randu();
        }
        
        void operator ()(const fmat &input, fmat &output, bool transpose=false) {
            output = linear->weight*input + bias;
        }

        ssize_t<1> input_size() { return linear->weight.n_cols; }
        ssize_t<1> output_size() { return linear->weight.n_rows; }
        parameter_list get_parameters() { return linear->get_parameters(); }
    };
    
    /*class ReshapeOp {
    public:
        std::size_t input_width, input_height;
        std::size_t output_width, output_height;
    public:
        ReshapeOp(std::size_t input_width, std::size_t input_height,
                  std::size_t output_width, std::size_t output_height):
            input_width(input_width),
            input_height(input_height),
            output_width(output_width),
            output_height(output_height) {}
        
        void operator ()(const fmat &input, fmat &output) {
            output.reshape(output_width, output_height);
        }
        
//        std::size_t input_size() {}
//        std::size_t output_size() {}
    };*/
    
    class SpatialConvolutionOp {
    public:
        ssize_t<2> input_size_;
        fmat kernel;
    public:
        SpatialConvolutionOp(ssize_t<2> input_size, ssize_t<2> kernel_size);
        
        void operator()(const fmat &input, fmat &output);

        ssize_t<2> input_size() { return input_size_; }
        
        // calculate reduction: w =
        ssize_t<2> output_size() { return {0, 0}; }
    };
    
    class SigmoidOp {
    protected:
        ssize_t<1> size;
    public:
        SigmoidOp(ssize_t<1> input_size): size(input_size) {}
        
        void operator ()(const fvec &input, fvec &output) {
            output = 1.0 / (1.0 + exp(-input));
        }
        
        void operator ()(const fmat &input, fmat &output) {
            output = 1.0 / (1.0 + exp(-input));
        }
        
        ssize_t<1> input_size() { return size; }
        ssize_t<1> output_size() { return size; }
        parameter_list get_parameters();
    };
}

#pragma GCC visibility pop

#endif
