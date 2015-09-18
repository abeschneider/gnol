//
//  main.cpp
//  rae
//
//  Created by Abe Schneider on 9/17/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#include <iostream>
#include <armadillo>

#include "module.hpp"
#include "criterion.hpp"
#include "check_gradient.hpp"
#include "linear.hpp"
#include "sequence.hpp"
#include "concat.hpp"
#include "reshape.hpp"
#include "activation.hpp"

struct word {
    fvec rep;
    std::unique_ptr<word> left;
};

typedef std::list<word> sentence;

int main(int argc, const char * argv[]) {
    
    
    return 0;
}
