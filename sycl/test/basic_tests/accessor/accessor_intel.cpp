// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------accessor_121_enums.cpp - SYCL 1.2.1 accessor enum test
//---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <type_traits>

namespace sycl {
using namespace cl::sycl;
}

constexpr int N = 32;

int main() {
  // Check shorter enum names defined by Intel extension

  sycl::buffer<int> B(N);
  sycl::image<2> I(sycl::image_channel_order::rgba,
                   sycl::image_channel_type::fp32, sycl::range<2>(16, 16));

  // Test host access::target enums
  {
    sycl::accessor<int, 1, sycl::access_mode::write,
                   sycl::target::host_buffer>
        a1(B);

    sycl::accessor<sycl::float4, 2, sycl::access_mode::write,
                   sycl::target::host_image>
        a2(I); // <== This line causes segfault to start appearing
  }

  sycl::queue Q{};

  Q.submit([&](sycl::handler &h) {
    // Test access::mode enums
    {
      sycl::accessor<int, 1, sycl::access_mode::write,
                     sycl::target::global_buffer>
          a1(B, h);

      sycl::accessor<int, 1, sycl::access_mode::discard_write,
                     sycl::target::global_buffer>
          a2(B, h);

      sycl::accessor<int, 1, sycl::access_mode::discard_read_write,
                     sycl::target::global_buffer>
          a3(B, h);

      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::target::global_buffer>
          a4(B, h);

      sycl::accessor<int, 1, sycl::access_mode::read,
                     sycl::target::global_buffer>
          a5(B, h);

      sycl::accessor<int, 1, sycl::access_mode::atomic,
                     sycl::target::global_buffer>
          a6(B, h);
    }

    // Test device access::target enums
    {
      sycl::accessor<int, 1, sycl::access_mode::read,
                     sycl::target::constant_buffer>
          a7(B, h);

      sycl::accessor<sycl::float4, 2, sycl::access_mode::write,
                     sycl::target::image>
          a8(I, h); // <== This line causes segfault to start appearing

      sycl::accessor<sycl::float4, 1, sycl::access_mode::write,
                     sycl::target::image_array>
          a9(I, h); // <== This line causes segfault to start appearing


      sycl::accessor<int, 0, sycl::access_mode::read_write,
                     sycl::target::local>
          a10(h);

      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::target::local>
          a11(1024, h);
    }

    // Check that target template param has default of global
    static_assert(
        std::is_same<
            decltype(sycl::accessor<int, 1, sycl::access_mode::write>{B, h}),
            decltype(sycl::accessor<int, 1, sycl::access_mode::write,
                                    sycl::target::global_buffer>{
                B, h})>::value,
        "Default accessor template arg doesn't match expectation");

    h.parallel_for<class kern>(sycl::range<1>(N), [=](sycl::id<1> idx) {
      // Empty kernel
    });
  });

  return 0;
}
