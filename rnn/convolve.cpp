//
//  convolve.cpp
//  rnn
//
//  Created by Abe Schneider on 9/15/15.
//  Copyright (c) 2015 Abraham Schneider. All rights reserved.
//

#include "convolve.hpp"

using namespace gnol;

void convolve2d(const fmat &input, const fmat &kernel, fmat &output) {
    ssize_t<2> center({input.n_rows, input.n_cols});
    
    for (std::uint64_t i = 0; i < input.n_rows; i++) {
        for (std::uint64_t j = 0; j < input.n_cols; j++) {
            for (std::uint64_t m = 0; m < kernel.n_rows; m++) {
                // row index of flipped kernel
                std::size_t fm = kernel.n_rows -1 - m;
                
                for (std::size_t n=0; n < kernel.n_cols; n++) {
                    std::size_t fn = kernel.n_cols - 1 - n;
                    
                    std::int64_t ii = i + (m - center[0]);
                    std::int64_t jj = j + (n - center[1]);
                    
                    // make sure we are within bounds
                    if (ii >= 0 && ii < input.n_rows && jj >= 0 && jj < input.n_cols) {
                        output(i, j) += input(ii, jj) * kernel(fm, fn);
                    }
                }
            }
        }
    }
}

