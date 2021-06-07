//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

constexpr char const* all_reduce_direct_basename = "/test/all_reduce_direct/";

void test_one_shot_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = hpx::get_locality_id();

        hpx::future<std::uint32_t> overall_result =
            hpx::collectives::all_reduce(all_reduce_direct_basename, value,
                std::plus<std::uint32_t>{}, num_localities, i);

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

void test_multiple_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    auto all_reduce_direct_client = hpx::collectives::create_communicator(
        all_reduce_direct_basename, num_localities);

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = hpx::get_locality_id();

        hpx::future<std::uint32_t> overall_result =
            hpx::collectives::all_reduce(
                all_reduce_direct_client, value, std::plus<std::uint32_t>{});

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

int hpx_main()
{
    test_one_shot_use();
    test_multiple_use();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}
#endif
