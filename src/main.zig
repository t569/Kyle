const std = @import("std");
const kernel = @import("kernel.zig");
const root = @import("root.zig");
pub fn main() !void {
    var gpa = std.heap.page_allocator;
    var dataset = try root.getDummyDataset(&gpa);
    try root.normalizeDataset(&dataset);

    const C: f64 = 1.0;
    const epsilon: f64 = 0.1;
    const max_iter: usize = 100;

    const params = kernel.KernelParams{ .gamma = 0.5 };
    const kernel_ = kernel.rbfKernel;

    const result = try kernel.trainSMO(dataset.inputs, dataset.targets, C, epsilon, kernel_, params, max_iter, dataset.allocator);
    const alpha = result.alpha;
    const alpha_star = result.alpha_star;
    const bias = result.bias;
    kernel.evaluateModel(dataset, alpha, alpha_star, bias, kernel_, params);

    // free the allocated data
    dataset.allocator.free(alpha);
    dataset.allocator.free(alpha_star);
}
