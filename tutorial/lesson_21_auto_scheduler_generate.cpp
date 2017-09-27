// Halide tutorial lesson 21: Auto-Scheduler

// This lesson demonstrates how to use the auto-scheduler to generate a
// copy-pastable CPU schedule that can be subsequently improved upon.

// On linux or os x, you can compile and run it like so:

// g++ lesson_21_auto_scheduler_generate.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_21_generate
// export LD_LIBRARY_PATH=../bin   # For linux
// export DYLD_LIBRARY_PATH=../bin # For OS X
// ./lesson_21_generate -o . -f conv_layer target=host
// g++ lesson_21_auto_scheduler_run.cpp brighten_*.o -ldl -lpthread -o lesson_21_run
// ./lesson_21_run

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_21_auto_scheduler_run
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// We will define a generator for a convolution layer.
class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Input<Buffer<float>>  bias{"bias", 1};

    Output<Buffer<float>> f_ReLU{"ReLU", 4};

    void generate() {
        RDom r(filter.dim(0).min(), filter.dim(0).extent(),
               filter.dim(1).min(), filter.dim(1).extent(),
               filter.dim(2).min(), filter.dim(2).extent());

        f_conv(x, y, z, n) = bias(z);
        f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);
        f_ReLU(x, y, z, n) = max(0, f_conv(x, y, z, n));
    }

    void schedule() {
        // Provide estimates on the input image
        input.dim(0).set_bounds_estimate(0, 131);
        input.dim(1).set_bounds_estimate(0, 131);
        input.dim(2).set_bounds_estimate(0, 64);
        input.dim(3).set_bounds_estimate(0, 4);

        filter.dim(0).set_bounds_estimate(0, 3);
        filter.dim(1).set_bounds_estimate(0, 3);
        filter.dim(2).set_bounds_estimate(0, 64);
        filter.dim(3).set_bounds_estimate(0, 64);

        bias.dim(0).set_bounds_estimate(0, 64);

        // Provide estimates on the pipeline output f_ReLU
        f_ReLU.estimate(x, 0, 128)
            .estimate(y, 0, 128)
            .estimate(z, 0, 64)
            .estimate(n, 0, 4);

        // Auto-schedule the pipeline
        Pipeline p(f_ReLU);
        p.auto_schedule(target);
    }
private:
    Var x{"x"}, y{"y"}, z{"z"}, n{"n"};
    Func f_conv;
};

// As in lesson 15, we register our generator and then compile this
// file along with tools/GenGen.cpp.
HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)

// After compiling this file, see how to use it in
// lesson_21_auto_scheduler_run.cpp
