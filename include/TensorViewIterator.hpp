#pragma once

#include <iterator>
#include <vector>
#include "Utility.hpp"

template <typename T>
class TensorView;

template <typename T>
class TensorViewIterator
{
protected:
    std::vector<T> *data_;

    const std::vector<size_t> *shape_;

    const std::vector<size_t> *strides_;

    std::vector<size_t> indexes_;

    size_t index_;

public:
    friend class TensorView<T>;

    using traits = std::iterator_traits<T *>;
    using iterator_type = T;
    using iterator_category = typename traits::iterator_category;
    using value_type = typename traits::value_type;
    using difference_type = typename traits::difference_type;
    using reference = typename traits::reference;
    using pointer = typename traits::pointer;

    TensorViewIterator(std::vector<T> *data,
                       const std::vector<size_t> *shape,
                       const std::vector<size_t> *strides,
                       size_t index) :
        data_(data), shape_(shape), strides_(strides), indexes_(shape->size(), 0), index_(index)
    {
    }

    TensorViewIterator(const TensorViewIterator &iterator) :
        data_(iterator.data_), shape_(iterator.shape_), strides_(iterator.strides_),
        indexes_(iterator.indexes_), index_(iterator.index_)
    {
    }

    TensorViewIterator(TensorViewIterator &&iterator) = delete;

    ~TensorViewIterator() = default;

    auto operator=(TensorViewIterator iterator) -> TensorViewIterator &
    {
        data_ = iterator.data_;
        shape_ = iterator.shape_;
        strides_ = iterator.strides_;
        indexes_ = iterator.indexes_;
        index_ = iterator.index_;
        return *this;
    }

    auto operator=(TensorViewIterator &&iterator) -> TensorViewIterator & = delete;

    auto operator==(const TensorViewIterator<T> &iterator) const -> bool
    {
        return data_ == iterator.data_ && index_ == iterator.index_;
    }

    auto operator!=(const TensorViewIterator<T> &iterator) const -> bool
    {
        return !operator==(iterator);
    }

    auto operator++() -> TensorViewIterator<T> &
    {
        size_t shape_index = 0;
        for (; shape_index < shape_->size(); ++shape_index)
        {
            if (indexes_[shape_index] + 1 >= (*shape_)[shape_index])
            {
                index_ -= indexes_[shape_index] * (*strides_)[shape_index];
                indexes_[shape_index] = 0;
            }
            else
            {
                break;
            }
        }
        if (shape_index < shape_->size())
        {
            index_ += (*strides_)[shape_index];
            ++indexes_[shape_index];
        }
        else
        {
            index_ = product(*shape_);
        }
        return *this;
    }

    auto operator--() -> TensorViewIterator<T> &
    {
        size_t shape_index = 0;
        for (; shape_index < shape_->size(); ++shape_index)
        {
            if (indexes_[shape_index] == 0)
            {
                indexes_[shape_index] = (*shape_)[shape_index] - 1;
                index_ += indexes_[shape_index] * (*strides_)[shape_index];
            }
            else
            {
                break;
            }
        }
        if (shape_index < shape_->size())
        {
            --indexes_[shape_index];
            index_ -= (*strides_)[shape_index];
        }
        else
        {
            index_ = product(*shape_);
        }
        return *this;
    }

    auto operator+=(const size_t &idx) -> TensorViewIterator<T> &
    {
        // TODO(Majoneza): Do better
        for (size_t i = 0; i < idx; ++i)
        {
            operator++();
        }
        return *this;
    }

    auto operator-=(const size_t &idx) -> TensorViewIterator<T> &
    {
        // TODO(Majoneza): Do better
        for (size_t i = 0; i < idx; ++i)
        {
            operator--();
        }
        return *this;
    }

    [[nodiscard]] auto get_index() const -> size_t
    {
        return index_;
    }
};
