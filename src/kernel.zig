// the kernel functions
// computes inner products within higher dim space s=without changing the value
const core = @import("root.zig");
const std = @import("std");
const TrainResult = struct { alpha: []f64, alpha_star: []f64, bias: f64 };

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
) !TrainResult {
    const N = inputs.len;

    const alpha = try allocator.alloc(f64, N);
    const alpha_star = try allocator.alloc(f64, N);

    @memset(alpha, 0.0);
    @memset(alpha_star, 0.0);
    var bias: f64 = 0.0;

    for (0..maxIter) |_| {
        // precompute the kernel matrix
        std.debug.print("Precomputing kernel matrix...\n", .{});
        var K = try allocator.alloc([]f64, N);

        for (0..N) |i| {
            K[i] = try allocator.alloc(f64, N);

            for (0..N) |j| {
                K[i][j] = kernel(inputs[i], inputs[j], params);
                // std.debug.print("i: {d}, j: {d}\n", .{ i, j });
            }
        }

        // compute the error for each data point
        for (0..N) |i| {
            var f_i: f64 = bias;

            for (0..N) |j| {
                f_i += (alpha[j] - alpha_star[j]) * K[j][i];
            }
            const E_i = f_i - targets[i];

            std.debug.print("Sample {}: f_i = {:.3}, y_i = {:.3}, E_i = {:.3}\n", .{ i, f_i, targets[i], E_i });

            // now we need to check for KKT violation

            // none we skip
            if (!violatesKKT(alpha[i], alpha_star[i], E_i, epsilon, C)) continue;

            // violation we do these things
            std.debug.print("KKT violation at index {}\n", .{i});

            // choose j != i
            var j = (i + 1) % N; // simple heuristic
            if (j == i) j = (i + 2) % N; // avoid selecting i

            // compute E_j
            var f_j: f64 = bias;

            for (0..N) |k| {
                f_j += (alpha[k] - alpha_star[k]) * K[k][j];
            }

            const E_j = f_j - targets[j];

            // compute eta = K_ii + K_jj - 2K_ij
            const eta = K[i][i] + K[j][j] - 2.0 * K[i][j];
            std.debug.print("eta: {}\n", .{eta});

            if (eta <= 0.0) {
                std.debug.print("Skipping i={} and j={} due to non-positive eta\n", .{ i, j });
                continue;
            }

            // update alpha[i]
            const alpha_i_old = alpha[i];
            const alpha_star_i_old = alpha_star[i];
            const alpha_j_old = alpha[j];
            const alpha_star_j_old = alpha_star[j];

            alpha[i] += (E_j - E_i) / eta;
            alpha_star[i] += (E_i - E_j) / eta;
            alpha[j] += (E_i - E_j) / eta;
            alpha_star[j] += (E_j - E_i) / eta;

            // clip to [0, C]
            alpha[i] = @min(C, @max(0.0, alpha[i]));
            alpha_star[i] = @min(C, @max(0.0, alpha_star[i]));
            alpha[j] = @min(C, @max(0.0, alpha[j]));
            alpha_star[j] = @min(C, @max(0.0, alpha_star[j]));

            // update bias
            const delta_i = (alpha[i] - alpha_star[i]) - (alpha_i_old - alpha_star_i_old);
            const delta_j = (alpha[j] - alpha_star[j]) - (alpha_j_old - alpha_star_j_old);

            const b1 = bias - E_i - delta_i * K[i][i] - delta_j * K[j][i];
            const b2 = bias - E_j - delta_i * K[i][j] - delta_j * K[j][j];

            if (alpha[i] > 0.0 and alpha[i] < C) {
                bias = b1;
            } else if (alpha[j] > 0.0 and alpha[j] < C) {
                bias = b2;
            } else {
                bias = (b1 + b2) / 2.0;
            }

            std.debug.print("Updated alpha[{}] from {:.3} to {:.3}, alpha_star[{}] from {:.3} to {:.3}\n ======================\n", .{ i, alpha_i_old, alpha[i], i, alpha_star_i_old, alpha_star[i] });
        }
    }
    return TrainResult{ .alpha = alpha, .alpha_star = alpha_star, .bias = bias };
}

fn violatesKKT(alpha_i: f64, alpha_star_i: f64, E_i: f64, epsilon: f64, C: f64) bool {
    return @abs(E_i) > epsilon and (alpha_i < C or alpha_star_i < C);
}
