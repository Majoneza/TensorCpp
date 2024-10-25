#pragma once

#include <ranges>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <ostream>
#include "TensorViewIteratorImpl.hpp"
#include "TensorViewIteratorImplConst.hpp"
#include "TensorViewAxisIteratorImpl.hpp"
#include "TensorViewAxisIteratorImplConst.hpp"
#include "Utility.hpp"

template <typename T>
class Tensor;

template <typename T>
class TensorView
{
protected:
    std::vector<size_t> shape_;

    std::vector<size_t> strides_;

    std::vector<T> *data_;

    void setup_strides()
    {
        size_t count = 1;
        strides_.reserve(shape_.size());
        for (unsigned long &value : std::ranges::reverse_view(shape_))
        {
            strides_.push_back(count);
            count *= value;
        }
        std::ranges::reverse(strides_);
    }

    void dot_recursive(const TensorView<T> &tensor,
                       TensorView<T> &output,
                       size_t shape_index,
                       size_t first_index,
                       size_t second_index,
                       size_t output_index) const
    {
        if (shape_index == output.shape_.size())
        {
            T result = 0;
            for (size_t i = 0; i < shape_.back(); ++i)
            {
                result += data_->operator[](first_index) * tensor.data_->operator[](second_index);
                first_index += strides_.back();
                second_index += tensor.strides_[0];
            }
            output.data_->operator[](output_index) = result;
        }
        else
        {
            for (size_t i = 0; i < output.shape_[shape_index]; ++i)
            {
                dot_recursive(tensor,
                              output,
                              shape_index + 1,
                              first_index,
                              second_index,
                              output_index);
                if (shape_index >= shape_.size() - 1)
                {
                    second_index += tensor.strides_[shape_index - (shape_.size() - 1) + 1];
                }
                else
                {
                    first_index += strides_[shape_index];
                }
                output_index += output.strides_[shape_index];
            }
        }
    }

    void dot_recursive(const TensorView<T> &tensor, TensorView<T> &output) const
    {
        if (shape_.back() != tensor.shape_.front())
        {
            throw std::invalid_argument("Invalid tensor shapes");
        }
        dot_recursive(tensor, output, 0, 0, 0, 0);
    }

    template <typename Fn>
    void index_recursive(const Fn &func, size_t shape_index, size_t index) const
    {
        if (shape_index == shape_.size())
        {
            func(index);
        }
        else
        {
            for (size_t i = 0; i < shape_[shape_index]; ++i)
            {
                index_recursive(func, shape_index + 1, index);
                index += strides_[shape_index];
            }
        }
    }

    template <typename Fn>
    void index_recursive(const Fn &func) const
    {
        index_recursive(func, 0, 0);
    }

    template <typename Fn>
    void index_recursive(size_t *indexes, const Fn &func, size_t shape_index, size_t index) const
    {
        if (shape_index == shape_.size())
        {
            func(index);
        }
        else
        {
            for (size_t i = 0; i < shape_[shape_index]; ++i)
            {
                index_recursive(indexes, func, shape_index + 1, index);
                indexes[shape_index] += strides_[shape_index];
                index += strides_[shape_index];
            }
        }
    }

    template <typename Fn>
    void index_recursive(size_t *indexes, const Fn &func) const
    {
        index_recursive(indexes, func, 0, 0);
    }

    template <typename Fn>
    void index_recursive(const TensorView<T> &tensor,
                         const Fn &func,
                         size_t shape_index,
                         size_t first_index,
                         size_t second_index) const
    {
        if (shape_index == shape_.size())
        {
            func(first_index, second_index);
        }
        else
        {
            for (size_t i = 0; i < shape_[shape_index]; ++i)
            {
                index_recursive(tensor, func, shape_index + 1, first_index, second_index);
                first_index += strides_[shape_index];
                second_index += tensor.strides_[shape_index];
            }
        }
    }

    template <typename Fn>
    void index_recursive(const TensorView<T> &tensor, const Fn &func) const
    {
        if (shape_.size() != tensor.shape_.size())
        {
            throw std::invalid_argument("Invalid tensor shapes");
        }
        index_recursive(tensor, func, 0, 0, 0);
    }

    auto broadcast_view(size_t rank,
                        std::initializer_list<size_t> tensor_shape) const -> TensorView<T>
    {
        std::vector<size_t> new_strides(tensor_shape.size(), 0);
        size_t index_old = 0;
        size_t index_new = 0;
        if (strides_.size() > rank)
        {
            index_old = strides_.size() - 1 - rank;
        }
        else
        {
            index_new = rank - strides_.size() + 1;
        }
        for (; index_old < strides_.size() && index_new < new_strides.size();
             ++index_old, ++index_new)
        {
            new_strides[index_new] = strides_[index_old];
        }
        return TensorView<T>(data_,
                             tensor_shape.begin(),
                             tensor_shape.end(),
                             std::make_move_iterator(new_strides.begin()),
                             std::make_move_iterator(new_strides.end()));
    }

    auto broadcast_view(std::initializer_list<size_t> indexes,
                        std::initializer_list<size_t> tensor_shape) const -> TensorView<T>
    {
        std::vector<size_t> new_strides;
        new_strides.reserve(tensor_shape.size());
        for (const auto &index : indexes)
        {
            new_strides.push_back(strides_[index]);
        }
        return TensorView<T>(data_,
                             tensor_shape.begin(),
                             tensor_shape.end(),
                             std::make_move_iterator(new_strides.begin()),
                             std::make_move_iterator(new_strides.end()));
    }

    auto axis_view(std::initializer_list<size_t> axis) const -> TensorView<T>
    {
        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        new_shape.reserve(axis.size());
        new_strides.reserve(axis.size());
        for (const auto &idx : axis)
        {
            new_shape.push_back(shape_[idx]);
            new_strides.push_back(strides_[idx]);
        }
        return TensorView<T>(data_,
                             std::make_move_iterator(new_shape.begin()),
                             std::make_move_iterator(new_shape.end()),
                             std::make_move_iterator(new_strides.begin()),
                             std::make_move_iterator(new_strides.end()));
    }

    auto expand_axis_view(size_t axis, size_t axis_shape) const -> TensorView<T>
    {
        auto new_shape = copy_and_insert<size_t>(shape_, axis, axis_shape);
        auto new_strides = copy_and_insert<size_t>(strides_, axis, 0);
        return TensorView<T>(data_,
                             std::make_move_iterator(new_shape.begin()),
                             std::make_move_iterator(new_shape.end()),
                             std::make_move_iterator(new_strides.begin()),
                             std::make_move_iterator(new_strides.end()));
    }

    template <typename It>
    auto index(const It begin, const It end) const -> size_t
    {
        size_t index = 0;
        auto strides_it = strides_.begin();
        for (auto it = begin; it != end; ++it)
        {
            index += (*it) * (*strides_it);
            ++strides_it;
        }
        return index;
    }

    void output_tensor(std::ostream &output_stream, size_t shape_index, size_t index) const
    {
        if (shape_index == shape_.size())
        {
            output_stream << data_->operator[](index);
        }
        else
        {
            output_stream << '[';
            for (size_t i = 0; i < shape_[shape_index]; ++i)
            {
                if (i != 0)
                {
                    output_stream << ',';
                }
                output_tensor(output_stream, shape_index + 1, index);
                index += strides_[shape_index];
            }
            output_stream << ']';
        }
    }

    void output_tensor(std::ostream &output_stream) const
    {
        output_tensor(output_stream, 0, 0);
    }

public:
    friend class Tensor<T>;

    explicit TensorView(std::vector<T> *data) : shape_(1, 0), strides_(1, 0), data_(data) {}

    template <typename ShIt>
    explicit TensorView(std::vector<T> *data, const ShIt shape_begin, const ShIt shape_end) :
        shape_(shape_begin, shape_end), data_(data)
    {
        setup_strides();
    }

    // Unsafe
    template <typename ShIt, typename StIt>
    explicit TensorView(std::vector<T> *data,
                        const ShIt shape_begin,
                        const ShIt shape_end,
                        const StIt strides_begin,
                        const StIt strides_end) :
        shape_(shape_begin, shape_end), strides_(strides_begin, strides_end), data_(data)
    {
    }

    TensorView(const TensorView<T> &tensor) :
        shape_(tensor.shape_), strides_(tensor.strides_), data_(tensor.data_)
    {
    }

    TensorView(TensorView<T> &&tensor) noexcept :
        shape_(std::move(tensor.shape_)), strides_(std::move(tensor.strides_)),
        data_(std::move(tensor.data_))
    {
    }

    virtual ~TensorView() = default;

    auto operator=(TensorView<T> tensor) -> TensorView<T> &
    {
        shape_ = tensor.shape_;
        strides_ = tensor.strides_;
        data_ = tensor.data_;
        return *this;
    }

    auto operator=(TensorView<T> &&tensor) noexcept -> TensorView<T> &
    {
        shape_ = std::move(tensor.shape_);
        strides_ = std::move(tensor.strides_);
        data_ = std::move(tensor.data_);
        return *this;
    }

    auto dot(const TensorView<T> &tensor) const -> Tensor<T>
    {
        std::vector<size_t> new_shape(shape_.begin(), shape_.end() - 1);
        new_shape.insert(new_shape.end(), tensor.shape_.begin() + 1, tensor.shape_.end());
        //
        Tensor<T> new_tensor(0,
                             std::make_move_iterator(new_shape.begin()),
                             std::make_move_iterator(new_shape.end()));
        dot_recursive(tensor, new_tensor);
        //
        return new_tensor;
    }

    [[nodiscard]] auto index(std::initializer_list<size_t> position) const -> size_t
    {
        return index(position.begin(), position.end());
    }

    template <typename Fn>
    auto transform_index(const Fn &func) const -> Tensor<T>
    {
        std::vector<T> t_data;
        t_data.reserve(product(shape_));
        index_recursive([&](const size_t &index) {
            t_data.push_back(func(index));
        });
        return Tensor<T>(std::make_move_iterator(t_data.begin()),
                         std::make_move_iterator(t_data.end()),
                         shape_.begin(),
                         shape_.end());
    }

    template <typename Fn>
    auto transform_index(const TensorView<T> &tensor, const Fn &func) const -> Tensor<T>
    {
        std::vector<T> t_data;
        t_data.reserve(product(shape_));
        index_recursive(tensor, [&](const size_t &first, const size_t &second) {
            t_data.push_back(func(first, second));
        });
        return Tensor<T>(std::make_move_iterator(t_data.begin()),
                         std::make_move_iterator(t_data.end()),
                         shape_.begin(),
                         shape_.end());
    }

    template <typename Fn>
    auto transform(const Fn &func) const -> Tensor<T>
    {
        return transform_index([&](const size_t &index) {
            return func(data_->operator[](index));
        });
    }

    template <typename Fn>
    auto transform(const TensorView<T> &tensor, const Fn &func) const -> Tensor<T>
    {
        return transform_index(tensor, [&](const size_t &first, const size_t &second) {
            return func(data_->operator[](first), tensor.data_->operator[](second));
        });
    }

    template <typename Fn>
    void transform_index_self(const Fn &func)
    {
        index_recursive([&](const size_t &index) {
            func(index);
        });
    }

    template <typename Fn>
    void transform_index_self(const TensorView<T> &tensor, const Fn &func)
    {
        index_recursive(tensor, [&](const size_t &first, const size_t &second) {
            func(first, second);
        });
    }

    template <typename Fn>
    void transform_self(const Fn &func)
    {
        return transform_index_self([&](const size_t &index) {
            return func(data_->operator[](index));
        });
    }

    template <typename Fn>
    void transform_self(const TensorView<T> &tensor, const Fn &func)
    {
        return transform_index_self([&](const size_t &first, const size_t &second) {
            return func(data_->operator[](first), tensor.data_->operator[](second));
        });
    }

    template <typename Fn>
    auto fold(T initial, const Fn &func) const -> T
    {
        index_recursive([&](const size_t &index) {
            func(data_->operator[](index), initial);
        });
        return initial;
    }

    template <typename Fn>
    auto fold_index(T initial, const Fn &func) const -> T
    {
        index_recursive([&](const size_t &index) {
            func(index, initial);
        });
        return initial;
    }

    template <typename Fn>
    auto fold_axis(size_t axis, T initial, const Fn &func) const -> Tensor<T>
    {
        auto new_shape = copy_except(shape_, axis);
        Tensor<T> new_tensor(initial,
                             std::make_move_iterator(new_shape.begin()),
                             std::make_move_iterator(new_shape.end()));
        auto view = new_tensor.expand_axis(axis);
        index_recursive(view, [&](const size_t &first, const size_t &second) {
            func(data_->operator[](first), view.data->operator[](second));
        });
        return new_tensor;
    }

    void reshape(std::initializer_list<size_t> tensor_shape)
    {
        shape_ = tensor_shape;
        auto prod = product(shape_);
        if (prod > data_->size())
        {
            data_->insert(data_->end(), prod - data_->size(), 0.0);
        }
        else if (prod < data_->size())
        {
            data_->resize(prod);
        }
    }

    auto clone() const -> Tensor<T>
    {
        return transform([](const T &value) {
            return value;
        });
    }

    template <size_t N, typename Fn>
    void for_each(std::array<size_t, N> indexes, const Fn &func)
    {
        if (N != shape_.size())
        {
            throw std::invalid_argument("Invalid index array size");
        }
        index_recursive(indexes.data(), [&](const size_t &index) {
            func(index);
        });
    }

    auto sum() const -> T
    {
        return fold(0, [](const T &value, T &acc) {
            acc += value;
        });
    }

    auto sum(size_t axis) const -> Tensor<T>
    {
        return fold_axis(axis, 0.0, [](const T &value, T &acc) {
            acc += value;
        });
    }

    void swap_axis()
    {
        std::swap(shape_.back(), shape_[shape_.size() - 2]);
        std::swap(strides_.back(), strides_[strides_.size() - 2]);
    }

    void swap_axis(size_t first, size_t second)
    {
        std::swap(shape_[first], shape_[second]);
        std::swap(strides_[first], strides_[second]);
    }

    void reverse_axis()
    {
        std::ranges::reverse(shape_);
        std::ranges::reverse(strides_);
    }

    auto transpose() -> TensorView<T>
    {
        TensorView<T> view(*this);
        view.swap_axis();
        return view;
    }

    auto transpose(size_t first, size_t second) -> TensorView<T>
    {
        TensorView<T> view(*this);
        view.swap_axis(first, second);
        return view;
    }

    [[nodiscard]] auto size() const -> size_t
    {
        return product(shape_);
    }

    [[nodiscard]] auto ndims() const -> size_t
    {
        return shape_.size();
    }

    [[nodiscard]] auto dim(size_t index) const -> size_t
    {
        return shape_[index];
    }

    [[nodiscard]] auto max_dim() const -> size_t
    {
        return *std::ranges::max_element(shape_);
    }

    [[nodiscard]] auto dims() const -> const std::vector<size_t> &
    {
        return shape_;
    }

    auto view() -> TensorView<T>
    {
        return TensorView<T>(*this);
    }

    auto view(std::initializer_list<size_t> axis) -> TensorView<T>
    {
        return axis_view(axis);
    }

    auto broadcast(size_t rank, std::initializer_list<size_t> tensor_shape) -> TensorView<T>
    {
        return broadcast_view(rank, tensor_shape);
    }

    auto broadcast(std::initializer_list<size_t> indexes,
                   std::initializer_list<size_t> tensor_shape) -> TensorView<T>
    {
        return broadcast_view(indexes, tensor_shape);
    }

    auto expand_axis(size_t axis, size_t axis_shape = 1) -> TensorView<T>
    {
        return expand_axis_view(axis, axis_shape);
    }

    auto begin() -> TensorViewIteratorImpl<T>
    {
        return TensorViewIteratorImpl<T>(data_, &shape_, &strides_, 0);
    }

    auto begin() const -> TensorViewIteratorImplConst<T>
    {
        return TensorViewIteratorImplConst<T>(data_, &shape_, &strides_, 0);
    }

    auto cbegin() const -> TensorViewIteratorImplConst<T>
    {
        return TensorViewIteratorImplConst<T>(data_, &shape_, &strides_, 0);
    }

    auto end() -> TensorViewIteratorImpl<T>
    {
        return TensorViewIteratorImpl<T>(data_, &shape_, &strides_, size());
    }

    auto end() const -> TensorViewIteratorImplConst<T>
    {
        return TensorViewIteratorImplConst<T>(data_, &shape_, &strides_, size());
    }

    auto cend() const -> TensorViewIteratorImplConst<T>
    {
        return TensorViewIteratorImplConst<T>(data_, &shape_, &strides_, size());
    }

    auto begin_axis(size_t axis) -> TensorViewAxisIteratorImpl<T>
    {
        return TensorViewAxisIteratorImpl<T>(data_, shape_[axis], strides_[axis], 0);
    }

    auto begin_axis(size_t axis,
                    std::initializer_list<size_t> position) -> TensorViewAxisIteratorImpl<T>
    {
        return TensorViewAxisIteratorImpl<T>(data_, shape_[axis], strides_[axis], index(position));
    }

    auto begin_axis(size_t axis) const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, shape_[axis], strides_[axis], 0);
    }

    auto begin_axis(size_t axis,
                    std::initializer_list<size_t> position) const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, shape_[axis], strides_[axis], index(position));
    }

    auto cbegin_axis(size_t axis) const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, shape_[axis], strides_[axis], 0);
    }

    auto cbegin_axis(size_t axis,
                    std::initializer_list<size_t> position) const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, shape_[axis], strides_[axis], index(position));
    }

    auto end_axis() -> TensorViewAxisIteratorImpl<T>
    {
        return TensorViewAxisIteratorImpl<T>(data_, 0, 0, size());
    }

    auto end_axis() const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, 0, 0, size());
    }

    auto cend_axis() const -> TensorViewAxisIteratorImplConst<T>
    {
        return TensorViewAxisIteratorImplConst<T>(data_, 0, 0, size());
    }

    auto operator[](size_t index) -> T &
    {
        return data_->operator[](index);
    }

    auto operator[](size_t index) const -> const T &
    {
        return data_->operator[](index);
    }

    auto operator()(std::initializer_list<size_t> position) -> T &
    {
        return data_->operator[](index(position.begin(), position.end()));
    }

    auto operator()(std::initializer_list<size_t> position) const -> const T &
    {
        return data_->operator[](index(position.begin(), position.end()));
    }

    auto operator()(const TensorViewIterator<T> &iter) -> T &
    {
        return data_->operator[](iter.index_);
    }

    auto operator()(const TensorViewIterator<T> &iter) const -> const T &
    {
        return data_->operator[](iter.index_);
    }

    auto operator()(const TensorViewAxisIterator<T> &iter) -> T &
    {
        return data_->operator[](iter.index_);
    }

    auto operator()(const TensorViewAxisIterator<T> &iter) const -> const T &
    {
        return data_->operator[](iter.index_);
    }

    friend auto operator+(const TensorView<T> &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return lhs.transform(rhs, [](const T &first, const T &second) {
            return first + second;
        });
    }

    friend auto operator+(const TensorView<T> &lhs, const T &rhs) -> Tensor<T>
    {
        return lhs.transform([=](const T &first) {
            return first + rhs;
        });
    }

    friend auto operator+(const T &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return rhs.transform([=](const T &value) {
            return lhs + value;
        });
    }

    auto operator+=(const T &value) -> TensorView<T> &
    {
        transform_self([=](T &first) {
            first += value;
        });
        return *this;
    }

    auto operator+=(const TensorView<T> &tensor) -> TensorView<T> &
    {
        transform_self(tensor, [](T &first, const T &second) {
            first += second;
        });
        return *this;
    }

    friend auto operator-(const TensorView<T> &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return lhs.transform(rhs, [](const T &first, const T &second) {
            return first - second;
        });
    }

    friend auto operator-(const TensorView<T> &lhs, const T &rhs) -> Tensor<T>
    {
        return lhs.transform([=](const T &first) {
            return first - rhs;
        });
    }

    friend auto operator-(const T &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return rhs.transform([=](const T &value) {
            return lhs - value;
        });
    }

    auto operator-=(const T &value) -> TensorView<T> &
    {
        transform_self([=](T &first) {
            first -= value;
        });
        return *this;
    }

    auto operator-=(const TensorView<T> &tensor) -> TensorView<T> &
    {
        transform_self(tensor, [](T &first, const T &second) {
            first -= second;
        });
        return *this;
    }

    friend auto operator*(const TensorView<T> &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return lhs.transform(rhs, [](const T &first, const T &second) {
            return first * second;
        });
    }

    friend auto operator*(const TensorView<T> &lhs, const T &rhs) -> Tensor<T>
    {
        return lhs.transform([=](const T &first) {
            return first * rhs;
        });
    }

    friend auto operator*(const T &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return rhs.transform([=](const T &value) {
            return lhs * value;
        });
    }

    auto operator*=(const T &value) -> TensorView<T> &
    {
        transform_self([=](T &first) {
            first *= value;
        });
        return *this;
    }

    auto operator*=(const TensorView<T> &tensor) -> TensorView<T> &
    {
        transform_self(tensor, [](T &first, const T &second) {
            first *= second;
        });
        return *this;
    }

    friend auto operator/(const TensorView<T> &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return lhs.transform(rhs, [](const T &first, const T &second) {
            return first / second;
        });
    }

    friend auto operator/(const TensorView<T> &lhs, const T &rhs) -> Tensor<T>
    {
        return lhs.transform([=](const T &first) {
            return first / rhs;
        });
    }

    friend auto operator/(const T &lhs, const TensorView<T> &rhs) -> Tensor<T>
    {
        return rhs.transform([=](const T &value) {
            return lhs / value;
        });
    }

    auto operator/=(const T &value) -> TensorView<T> &
    {
        transform_self([=](T &first) {
            first /= value;
        });
        return *this;
    }

    auto operator/=(const TensorView<T> &tensor) -> TensorView<T> &
    {
        transform_self(tensor, [](T &first, const T &second) {
            first /= second;
        });
        return *this;
    }

    friend auto operator<<(std::ostream &output_stream,
                           const TensorView<T> &tensor) -> std::ostream &
    {
        tensor.output_tensor(output_stream);
        return output_stream;
    }
};
