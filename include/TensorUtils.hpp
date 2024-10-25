#pragma once

#include "Tensor.hpp"

template <typename T>
auto one_hot(size_t index, size_t size) -> Tensor<T>
{
    std::vector<T> data(size, 0.0);
    data[index] = 1.0;
    return Tensor<T>(std::make_move_iterator(data.begin()),
                     std::make_move_iterator(data.end()),
                     {size});
}

template <typename T>
auto one_hot(size_t index, size_t size, std::initializer_list<size_t> shape) -> Tensor<T>
{
    std::vector<T> data(size, 0.0);
    data[index] = 1.0;
    return Tensor<T>(std::make_move_iterator(data.begin()),
                     std::make_move_iterator(data.end()),
                     shape);
}
