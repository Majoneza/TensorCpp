#pragma once

#include "TensorView.hpp"
#include <numeric>

template <typename T>
class Tensor : public TensorView<T>
{
protected:
    std::vector<T> raw_data_;

public:
    static auto vector(std::initializer_list<T> tensor_data) -> Tensor<T>
    {
        std::vector<size_t> shape = {tensor_data.size()};
        std::vector<T> raw_data;
        raw_data.reserve(tensor_data.size());
        for (auto it = tensor_data.begin(); it != tensor_data.end(); ++it)
        {
            raw_data.push_back(*it);
        }
        return Tensor(std::make_move_iterator(raw_data.begin()),
                      std::make_move_iterator(raw_data.end()),
                      std::make_move_iterator(shape.begin()),
                      std::make_move_iterator(shape.end()));
    }

    static auto matrix(std::initializer_list<std::initializer_list<T>> tensor_data) -> Tensor<T>
    {
        auto min_size = std::numeric_limits<size_t>::max();
        for (auto &value : tensor_data)
        {
            if (value.size() < min_size)
            {
                min_size = value.size();
            }
        }
        if (min_size == std::numeric_limits<size_t>::max())
        {
            min_size = 0;
        }
        std::vector<size_t> shape = {tensor_data.size(), min_size};
        std::vector<T> raw_data;
        raw_data.reserve(tensor_data.size() * min_size);
        for (auto &value : tensor_data)
        {
            auto iter = value.begin();
            for (size_t i = 0; i < min_size; ++i, ++iter)
            {
                raw_data.push_back(*iter);
            }
        }
        return Tensor(std::make_move_iterator(raw_data.begin()),
                      std::make_move_iterator(raw_data.end()),
                      std::make_move_iterator(shape.begin()),
                      std::make_move_iterator(shape.end()));
    }

    Tensor() : TensorView<T>(&raw_data_), raw_data_() {}

    explicit Tensor(std::initializer_list<T> tensor_data,
                    std::initializer_list<size_t> tensor_shape) :
        Tensor(tensor_data.begin(), tensor_data.end(), tensor_shape.begin(), tensor_shape.end())
    {
    }

    template <std::ranges::input_range Range>
    explicit Tensor(Range &&range, std::initializer_list<size_t> tensor_shape) :
        Tensor(std::ranges::begin(range),
               std::ranges::end(range),
               tensor_shape.begin(),
               tensor_shape.end())
    {
    }

    explicit Tensor(const T &value, std::initializer_list<size_t> tensor_shape) :
        Tensor(value, tensor_shape.begin(), tensor_shape.end())
    {
    }

    template <typename DtIt>
    explicit Tensor(const DtIt data_begin,
                    const DtIt data_end,
                    std::initializer_list<size_t> tensor_shape) :
        Tensor(data_begin, data_end, tensor_shape.begin(), tensor_shape.end())
    {
    }

    template <typename ShIt>
    explicit Tensor(const T &value, const ShIt shape_begin, const ShIt shape_end) :
        TensorView<T>(&raw_data_, shape_begin, shape_end), raw_data_(product(this->shape_), value)
    {
    }

    template <typename DtIt, typename ShIt>
    explicit Tensor(const DtIt data_begin,
                    const DtIt data_end,
                    const ShIt shape_begin,
                    const ShIt shape_end) :
        TensorView<T>(&raw_data_, shape_begin, shape_end), raw_data_(data_begin, data_end)
    {
        const auto expected_size = product(this->shape_);
        raw_data_.resize(expected_size, 0);
    }

    // Unsafe
    template <typename DtIt, typename ShIt, typename StIt>
    explicit Tensor(const DtIt data_begin,
                    const DtIt data_end,
                    const ShIt shape_begin,
                    const ShIt shape_end,
                    const StIt strides_begin,
                    const StIt strides_end) :
        TensorView<T>(&raw_data_, shape_begin, shape_end, strides_begin, strides_end),
        raw_data_(data_begin, data_end)
    {
    }

    explicit Tensor(const TensorView<T> &tensor) :
        TensorView<T>(&raw_data_,
                      tensor.shape.begin(),
                      tensor.shape.end(),
                      tensor.strides.begin(),
                      tensor.strides.end()),
        raw_data_(*tensor.data)
    {
    }

    Tensor(const Tensor<T> &tensor) :
        TensorView<T>(&raw_data_,
                      tensor.shape.begin(),
                      tensor.shape.end(),
                      tensor.strides.begin(),
                      tensor.strides.end()),
        raw_data_(tensor.raw_data_)
    {
    }

    Tensor(Tensor<T> &&tensor) noexcept :
        TensorView<T>(&raw_data_,
                      std::make_move_iterator(tensor.shape.begin()),
                      std::make_move_iterator(tensor.shape.end()),
                      std::make_move_iterator(tensor.strides.begin()),
                      std::make_move_iterator(tensor.strides.end())),
        raw_data_(std::move(tensor.raw_data_))
    {
    }

    ~Tensor() override = default;

    auto operator=(TensorView<T> tensor) -> Tensor<T> &
    {
        this->shape = tensor.shape;
        this->strides = tensor.strides;
        this->raw_data_ = *tensor.data;
        return *this;
    }

    auto operator=(Tensor<T> tensor) -> Tensor<T> &
    {
        this->shape = tensor.shape;
        this->strides = tensor.strides;
        this->raw_data_ = tensor.raw_data_;
        return *this;
    }

    auto operator=(Tensor<T> &&tensor) noexcept -> Tensor<T> &
    {
        this->shape = std::move(tensor.shape);
        this->strides = std::move(tensor.strides);
        this->raw_data_ = std::move(tensor.raw_data_);
        return *this;
    }
};
