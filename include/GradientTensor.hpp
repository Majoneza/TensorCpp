#pragma once

#include "Tensor.hpp"

template <typename T>
class GradientTensor : public Tensor<T>
{
protected:
    std::vector<T> gradient_data_;

    long operationId_;

public:
    explicit GradientTensor(const Tensor<T> &tensor) :
        Tensor<T>(tensor), gradient_data_(tensor.size(), 0),
        operationId_(random())
    {
    }
};
