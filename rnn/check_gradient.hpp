//
//  check_gradient.h
//  rnn
//
//  Created by Abe Schneider on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__check_gradient__
#define __rnn__check_gradient__

#include "module.hpp"

#include <list>
#include <memory>
#include <armadillo>

#pragma GCC visibility push(default)

#include "module.hpp"

namespace gnol {
    std::list<real_t> check_gradient(std::function<real_t (const vector_t &)> fn,
                                    GradientModule &mod,
                                    const vector_t &input,
                                    real_t eps);
}

#pragma GCC visibility pop

#endif /* defined(__rnn__check_gradient__) */
