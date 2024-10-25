#pragma once

#include <ranges>
#include <vector>

template <std::ranges::input_range Range>
constexpr auto product(Range &&range)
{
    std::ranges::range_value_t<Range> result = 1;
    for (auto it = std::ranges::begin(range); it != std::ranges::end(range); ++it)
    {
        result *= *it;
    }
    return result;
}

template <typename T>
auto copy_and_insert(const std::vector<T> &source, size_t index, const T &value) -> std::vector<T>
{
    std::vector<T> dest;
    dest.reserve(source.size() + 1);
    for (size_t i = 0; i < source.size(); ++i)
    {
        if (index == i)
        {
            dest.push_back(value);
        }
        dest.push_back(source[i]);
    }
    if (index == source.size())
    {
        dest.push_back(value);
    }
    return dest;
}

template <typename T>
auto copy_except(const std::vector<T> &source, size_t index) -> std::vector<T>
{
    std::vector<T> dest;
    dest.reserve(source.size() - 1);
    for (size_t i = 0; i < index; ++i)
    {
        dest.push_back(source[i]);
    }
    for (size_t i = index + 1; i < source.size(); ++i)
    {
        dest.push_back(source[i]);
    }
    return dest;
}
