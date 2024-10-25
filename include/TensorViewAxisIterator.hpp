#pragma once

#include <iterator>
#include <vector>

template <typename T>
class TensorView;

template <typename T>
class TensorViewAxisIterator
{
private:
    std::vector<T> *data_;

    size_t stride_;

    size_t max_index_;

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

    TensorViewAxisIterator(std::vector<T> *data, size_t shape, size_t stride, size_t index) :
        data_(data), stride_(stride), max_index_(shape * stride), index_(index)
    {
    }

    TensorViewAxisIterator(const TensorViewAxisIterator &iterator) :
        data_(iterator.data_), stride_(iterator.stride_), max_index_(iterator.max_index_),
        index_(iterator.index_)
    {
    }

    TensorViewAxisIterator(TensorViewAxisIterator &&iterator) = delete;

    ~TensorViewAxisIterator() = default;

    auto operator=(TensorViewAxisIterator iterator) -> TensorViewAxisIterator &
    {
        data_ = iterator.data_;
        stride_ = iterator.stride_;
        max_index_ = iterator.max_index_;
        index_ = iterator.index_;
        return *this;
    }

    auto operator=(TensorViewAxisIterator &&iterator) -> TensorViewAxisIterator & = delete;

    auto operator==(const TensorViewAxisIterator<T> &iterator) const -> bool
    {
        return data_ == iterator.data_ && index_ == iterator.index_;
    }

    auto operator!=(const TensorViewAxisIterator<T> &iterator) const -> bool
    {
        return !operator==(iterator);
    }

    auto operator++() -> TensorViewAxisIterator<T> &
    {
        index_ = std::min(max_index_, index_ + stride_);
        return *this;
    }

    auto operator--() -> TensorViewAxisIterator<T> &
    {
        index_ = std::min(static_cast<size_t>(0), index_ - stride_);
        return *this;
    }

    auto operator+=(const size_t &idx) -> TensorViewAxisIterator<T> &
    {
        index_ = std::min(max_index_, index_ + idx * stride_);
        return *this;
    }

    auto operator-=(const size_t &idx) -> TensorViewAxisIterator<T> &
    {
        index_ = std::max(static_cast<size_t>(0), index_ - idx * stride_);
        return *this;
    }

    [[nodiscard]] auto get_index() const -> size_t
    {
        return index_;
    }
};
