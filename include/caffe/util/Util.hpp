/*
 * Util.hpp
 *
 *  Created on: 8 Apr, 2014
 *      Author: alfredtofu
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

template<class bidiiter>
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}


#endif /* UTIL_HPP_ */
