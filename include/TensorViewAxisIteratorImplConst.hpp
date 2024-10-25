#pragma once

#include "TensorViewAxisIterator.hpp"

template <typename T>
class TensorViewAxisIteratorImplConst : public TensorViewAxisIterator<T>
{
    using Base = TensorViewAxisIterator<T>;

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
