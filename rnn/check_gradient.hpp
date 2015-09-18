//
//  check_gradient.h
//  rnn
//
//  Created by Lais Washington on 8/31/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#ifndef __rnn__check_gradient__
#define __rnn__check_gradient__

#include "layer.hpp"

#include <list>
#include <memory>
#include <armadillo>

#pragma GCC visibility push(default)

#include "module.hpp"

namespace gnol {
    std::list<float> check_gradient(std::function<float (const fvec &)> fn,
                                    GradientModule &mod,
                                    const fvec &input,
                                    float eps);
}

#pragma GCC visibility pop

#endif /* defined(__rnn__check_gradient__) */
