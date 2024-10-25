#pragma once

#include "TensorViewIterator.hpp"

template <typename T>
class TensorViewIteratorImplConst : public TensorViewIterator<T>
{
    using Base = TensorViewIterator<T>;

public:
    using Base::Base;

    auto operator*() const -> const Base::value_type &
    {
        return this->data_->operator[](this->index_);
    }

    auto operator->() const -> const Base::value_type *
    {
        return this->data_->operator[](this->index_);
    }
};
