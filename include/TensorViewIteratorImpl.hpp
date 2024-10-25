#pragma once

#include "TensorViewIterator.hpp"

template <typename T>
class TensorViewIteratorImpl : public TensorViewIterator<T>
{
    using Base = TensorViewIterator<T>;

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
