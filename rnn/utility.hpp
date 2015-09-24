//
//  utility.hpp
//  rnn
//
//  Created by Abe Schneider on 9/7/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef rnn_utility_hpp
#define rnn_utility_hpp

#include <list>
#include <array>

#include <boost/range.hpp>
#include <boost/optional.hpp>

#include <armadillo>

namespace gnol {
    using namespace arma;
    
    typedef mat matrix_t;
    typedef vec vector_t;
    typedef matrix_t::elem_type real_t;
    
    typedef std::list<boost::iterator_range<matrix_t::iterator>> parameter_list;
    
    extern parameter_list empty_parameter_list;
    
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
        
        ssize_t(std::array<std::size_t, D> values) {
            std::copy(values.begin(), values.end(), extent.begin());
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
        
        ssize_t(std::array<std::size_t, 1> values) {
            extent = values[0];
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
        
        size_t(std::initializer_list<std::size_t> &values) {
            extent.resize(values.size());
            std::copy(values.begin(), values.end(), extent.begin());
        }
        
        template <std::size_t D>
        size_t(ssize_t<D> size) {
            extent.resize(D);
            std::copy(size.begin(), size.end(), extent.begin());
        }
        
        size_t(std::size_t size) {
            extent.resize(1);
            extent[0] = size;
        }
        
        std::size_t num_elements() const {
            return std::accumulate(extent.begin(), extent.end(), 0,
               [](std::size_t a, std::size_t b) -> std::size_t {
                   return a+b;
               });
        }
        
        
        size_t &operator =(std::initializer_list<std::size_t> values) {
            extent.resize(values.size());
            std::copy(values.begin(), values.end(), extent.begin());
            return *this;
        }
        
        size_t &operator =(size_t &size) {
            extent.resize(size.dims());
            std::copy(size.begin(), size.end(), extent.begin());
            return *this;
        }
        
        std::size_t dims() const { return extent.size(); }
        std::size_t size() const { return extent.size(); }
        
        std::size_t &operator [](std::size_t index) { return extent[index]; }
        std::size_t operator [](std::size_t index) const { return extent[index]; }
        
        iterator begin() { return extent.begin(); }
        iterator end() { return extent.end(); }
        
        void write(std::ostream &os) const {
            os << "(";
            
            auto pos = extent.begin();
            while (pos != extent.end()) {
                os << *pos++;
                if (pos != extent.end())
                    os << ", ";
            }
            os << ")";
        }
        
        std::ostream &operator <<(std::ostream &os) {
            write(os);
            return os;
        }
        
        friend std::ostream &operator <<(std::ostream &os, size_t size) {
            size.write(os);
            return os;
        }
    };
    
    template <typename T>
    struct variable_storage {
        T *storage;
        
        variable_storage(const size_t &size) {
            storage = new T[size.num_elements()];
        }
        
        variable_storage(ssize_t<1> size) {
            storage = new T[size];
        }
        
        variable_storage(ssize_t<2> size) {
            storage = new T[size[0]*size[1]];
        }
        
        variable_storage(std::size_t num_elements) {
            storage = new T[num_elements];
        }
        
        variable_storage(std::size_t rows, std::size_t cols) {
            storage = new T[rows*cols];
        }
        
        ~variable_storage() {
            delete storage;
        }
        
        T *get() { return storage; }
    };
    
    template <typename MatrixT>
    class variable {
    public:
        typedef typename MatrixT::elem_type element_t;
    protected:
        bool shared;
        std::shared_ptr<MatrixT> value;
    protected:
        // used by shared variant
        variable(variable &v, bool):
            shared(true),
            value(v.value) {}
    public:
        variable(const MatrixT &value):
            shared(false),
            value(std::make_shared<MatrixT>(value)) {}
        
        variable(MatrixT &&value):
            shared(false),
            value(std::make_shared<MatrixT>(value)) {}
        
        variable(variable &v):
            shared(false),
            value(v.value) {}
        
        variable(variable &&v):
            shared(false),
            value(v.value) {}
        
        variable(size_t size):
            shared(false)
        {
            if (size.dims() == 1)
                value = std::make_shared<MatrixT>(size[0], 1);
            else
                value = std::make_shared<MatrixT>(size[0], size[1]);
        }
        
        template<typename eT=element_t, typename std::enable_if<std::is_convertible<MatrixT,Col<eT>>::value>::type...>
        variable(ssize_t<1> size):
            shared(false),
            value(std::make_shared<MatrixT>(size[0])) {}
        
        template<typename eT=element_t, typename std::enable_if<std::is_convertible<MatrixT,Mat<eT>>::value>::type...>
        variable(ssize_t<2> size):
            shared(false),
            value(std::make_shared<MatrixT>(size[0], size[1])) {}
        
        MatrixT &operator *() { return *value; }
        std::shared_ptr<MatrixT> operator ->() { return value; }
        
        const MatrixT &operator *() const { return *value; }
        std::shared_ptr<const MatrixT> operator ->() const { return value; }
    };
    
    template <typename MatrixT>
    class shared_variable: public variable<MatrixT> {
    public:
        shared_variable(variable<MatrixT> &v):
            variable<MatrixT>(v, true) {}
    };
    
    template <typename MatrixT>
    shared_variable<MatrixT> share(variable<MatrixT> &v) {
        return shared_variable<MatrixT>(v);
    }

    inline variable<vector_t> make_vector(ssize_t<1> size) {
        variable<vector_t> tmp(size);
        return tmp;
    }
    
    inline size_t size_l(std::initializer_list<std::size_t> lst) {
        return size_t(lst);
    }
    
//    template <std::size_t s, std::size_t...sizes>
//    ssize_t<sizeof...(size)> ssize_l(Args... args) {
//        return ssize_t<sizeof...(Args)>(args...);
//    }
    
    // TODO: review
    template<typename...T>
    std::array<std::size_t, 1+sizeof...(T)> make_array(std::size_t arg0, T...args) {
        return {{arg0, static_cast<std::size_t>(args)...}};
    }

    template <typename...Args>
    ssize_t<sizeof...(Args)> size(Args...args) {
        return ssize_t<sizeof...(Args)>(make_array(args...));
    }
    
    vector_t concat(std::initializer_list<vector_t> &&lst);
}

#endif
