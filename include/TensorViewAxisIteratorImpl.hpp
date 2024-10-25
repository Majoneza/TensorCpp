#pragma once

#include "TensorViewAxisIterator.hpp"

template <typename T>
class TensorViewAxisIteratorImpl : public TensorViewAxisIterator<T>
{
    using Base = TensorViewAxisIterator<T>;

public:
    using Base::Base;

    auto operator*() -> Base::reference
    {
        return this->data_->operator[](this->index_);
    }

    auto operator->() -> Base::pointer
    {
        return this->data_->operator[](this->index_);
    }
};
