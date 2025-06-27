// the kernel functions
// computes inner products within higher dim space s=without changing the value
const core = @import("root.zig");
const std = @import("std");

pub const KernelType = enum {
    Linear,
    Polynomial,
    RBF,
};

pub const KernelParams = struct {
    gamma: f64 = 1.0, // RBF/Polynomial
    coef0: f64 = 0.0, // Polynomial
    degree: u32 = 3, // Polynomial
};

const KernelFunction = fn (x: []const f64, y: []const f64, params: KernelParams) f64;
pub fn linearKernel(x: []const f64, y: []const f64, _: KernelParams) f64 {
    // K(x, y) = x.y
    var sum: f64 = 0.0;

    // sum the dot between x and y
    for (x, 0..) |x_i, i| {
        sum += x_i * y[i];
    }

    return sum;
}

pub fn polynomialKernel(x: []const f64, y: []const f64, p: KernelParams) f64 {
    //K(x,y)=(γ⋅x⋅y+c)^d
    const dot_product = linearKernel(x, y, p);
    return std.math.pow(f64, p.gamma * dot_product + p.coef0, @floatFromInt(p.degree));
}

pub fn rbfKernel(x: []const f64, y: []const f64, p: KernelParams) f64 {
    // K(x,y)=exp(−γ⋅∥x−y∥2)

    var sum_sq: f64 = 0.0;

    for (x, 0..) |x_i, i| {
        const diff = x_i - y[i];
        sum_sq += diff * diff;
    }

    return @exp(-p.gamma * sum_sq);
}

pub fn getKernelFunction(kernel_type: KernelType) KernelFunction {
    return switch (kernel_type) {
        .Linear => linearKernel,
        .Polynomial => polynomialKernel,
        .RBF => rbfKernel,
    };
}

// the loss function
pub fn epsilonInsensitiveLoss(predicted: f64, actual: f64, epsilon: f64) f64 {
    const error_term = @abs(predicted - actual);

    // this zeros all the value within the tolerance epsilon range
    return if (error_term <= epsilon) 0.0 else error_term - epsilon;
}

// the total loss function
pub fn totalLoss(
    weights: []const f64,
    bias: f64,
    dataset: core.Dataset,
    epsilon: f64,
    C: f64,
) f64 {
    var loss: f64 = 0.0;

    for (dataset.inputs, 0..) |x_i, i| {
        var prediction: f64 = bias;
        for (x_i, 0..) |x_j, j| {
            prediction += weights[j] * x_j;
        }

        // computes the loss for the current sample and adds it to the total loss
        const eps_loss = epsilonInsensitiveLoss(prediction, dataset.targets[i], epsilon);
        loss += eps_loss;
    }

    // Regularization term
    var reg: f64 = 0.0;
    for (weights) |w| reg += w * w;

    return 0.5 * reg + C * loss;
}

// lagrange multiplier function

// precompute the kernel matrix

fn buildKernelMatrix(
    inputs: []const []const f64, // cant this just be Vec? No Vec has an allocator
    kernel: KernelFunction,
    params: KernelParams,
    allocator: *std.mem.Allocator,
) ![][]f64 {
    const N = inputs.len;
    var K = try allocator.alloc([]f64, N);
    for (inputs, 0..) |x_i, i| {
        K[i] = try allocator.alloc(f64, N);

        for (inputs, 0..) |x_j, j| {
            K[i][j] = kernel(x_i, x_j, params);
        }
    }

    return K;
}

// objective function for the lagrange multipliers
// evaluate dual loss
fn dualObjective(
    alpha: []const f64,
    alpha_star: []const f64,
    y: []const f64,
    K: [][]f64,
    epsilon: f64,
) f64 {
    const N = alpha.len;
    var sum: f64 = 0.0;

    for (0..N) |i| {
        for (0..N) |j| {
            const coeff_i = alpha[i] - alpha_star[i];
            const coeff_j = alpha[j] - alpha_star[j];
            sum += coeff_i * coeff_j * K[i][j];
        }
    }

    var linear_term: f64 = 0.0;
    for (0..N) |i| {
        linear_term += y[i] * (alpha[i] - alpha_star[i]) - epsilon * (alpha[i] + alpha_star[i]);
    }

    return -0.5 * sum + linear_term;
}

// equality constraints for the lagrange multipliers

fn constraintSum(alpha: []const f64, alpha_star: []const f64) f64 {
    var sum: f64 = 0.0;

    for (alpha, 0..) |a, i| {
        sum += a - alpha_star[i];
    }
    return sum;
}

// SMO Sequential Minimal Optimization
// This is a simplified version of the SMO algorithm for training SVMs

pub fn predict(
    x: []const f64,
    inputs: []const []const f64,
    alpha: []const f64,
    alpha_star: []const f64,
    bias: f64,
    kernel: KernelFunction,
    params: KernelParams,
) f64 {
    var result = bias;

    for (inputs, 0..) |x_i, i| {
        const koef = kernel(x, x_i, params);
        result += (alpha[i] - alpha_star[i]) * koef;
    }

    return result;
}

// Evaluation
pub fn evaluateModel(
    dataset: core.Dataset,
    alpha: []const f64,
    alpha_star: []const f64,
    bias: f64,
    kernel: KernelFunction,
    params: KernelParams,
) void {
    var total_error: f64 = 0.0;
    for (dataset.inputs, 0..) |x, i| {
        const y_true = dataset.targets[i];
        const y_pred = predict(x, dataset.inputs, alpha, alpha_star, bias, kernel, params);
        const error_ = y_pred - y_true;
        std.debug.print("Sample {}: Prediction = {:.3}, Actual = {:.3}, Error = {:.3}\n", .{ i, y_pred, y_true, error_ });
        total_error += error_ * error_;
    }
    const mse = total_error / @as(f64, @floatFromInt(dataset.inputs.len));
    std.debug.print("\nMean Squared Error: {:.4}\n", .{mse});
}

// for debug purposes

fn get_writer() std.io.Writer(std.fs.File.Writer, std.io.StdOutStream) {
    return std.io.getStdOut().writer();
}

// the thorn in my side: training
pub fn trainSMO(
    inputs: [][]f64, // xᵢ
    targets: []f64, // yᵢ
    C: f64, // Regularization parameter
    epsilon: f64, // ε-tube width
    kernel: KernelFunction, // Kernel function
    params: KernelParams, // Kernel params
    maxIter: usize,
    allocator: *std.mem.Allocator,
) !void {
    const N = inputs.len;

    const alpha = try allocator.alloc(f64, N);
    const alpha_star = try allocator.alloc(f64, N);
    @memset(alpha, 0.0);
    @memset(alpha_star, 0.0);
    const bias: f64 = 0.0;

    _ = epsilon;
    _ = targets;
    _ = C;
    _ = maxIter;
    _ = bias;
    _ = kernel;
    _ = params;
}

const TrainResult = struct { alpha: []f64, alpha_star: []f64, bias: f64 };
