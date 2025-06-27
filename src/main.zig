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

    // const alpha = try dataset.allocator.alloc(f64, dataset.inputs.len);
    // const alpha_star = try dataset.allocator.alloc(f64, dataset.inputs.len);
    // @memset(alpha, 0.0);
    // @memset(alpha_star, 0.0);

    // const result = try kernel.trainSMO(dataset, C, epsilon, kernel_, params, max_iter);
    // const alpha = result.alpha;
    // const alpha_star = result.alpha_star;
    // const bias = result.bias;

    try kernel.trainSMO(dataset.inputs, dataset.targets, C, epsilon, kernel_, params, max_iter, dataset.allocator);
    // kernel.evaluateModel(dataset, alpha, alpha_star, bias, kernel_, params);
}
