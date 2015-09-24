//
//  utlity.cpp
//  rnn
//
//  Created by Abe Schneider on 9/17/15.
//  Copyright © 2015 Abraham Schneider. All rights reserved.
//

#include "utility.hpp"

namespace gnol {
    parameter_list empty_parameter_list;
    
    vector_t concat(std::initializer_list<vector_t> &&lst) {
        std::size_t size = 0;
        for (auto &v : lst) size += v.size();
        
        vector_t joined(size);
        vector_t::iterator out = joined.begin();
        for (auto &v : lst) out = std::copy(v.begin(), v.end(), out);
        
        return std::move(joined);
    }
}