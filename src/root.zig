//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

// the dataset format
pub const Vec = []f64;
pub const Dataset = struct {
    inputs: []Vec, // an array of our input vectors
    targets: []f64, // our y value predictions
    allocator: *std.mem.Allocator, // used to do throw away utitlities for our data, it has to be a page allocator unfortunately
};

// get a make shift dataset for now

pub fn getDummyDataset(allocator: *std.mem.Allocator) !Dataset {
    const inputs: [4][2]f64 = .{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
        .{ 4.0, 5.0 },
    };
    const targets: [4]f64 = .{ 2.1, 2.9, 3.8, 5.0 };

    const inputs_alloc = try allocator.alloc(Vec, inputs.len);
    const targets_alloc = try allocator.alloc(f64, targets.len);

    for (inputs, 0..) |input, i| {
        inputs_alloc[i] = try allocator.dupe(f64, &input);
        targets_alloc[i] = targets[i];
    }

    return Dataset{
        .inputs = inputs_alloc,
        .targets = targets_alloc,
        .allocator = allocator,
    };
}

pub fn normalizeDataset(dataset: *Dataset) !void {
    // xj = (xj - xj_min)/(xj_max - xj_min)

    // get the vector dimensions
    const num_features = dataset.inputs[0].len;

    var mins = try dataset.allocator.alloc(f64, num_features);
    var maxs = try dataset.allocator.alloc(f64, num_features);

    // for each element in the first vec, assign it to both max and min
    for (dataset.inputs[0], 0..) |val, j| {
        mins[j] = val;
        maxs[j] = val;
    }

    // now use max and min to find the maximum and minimum in the input data

    for (dataset.inputs) |vec| {
        for (vec, 0..) |val, j| {
            if (val < mins[j]) mins[j] = val;
            if (val > maxs[j]) maxs[j] = val;
        }
    }

    // finally we normalize
    // we take a pointer to the values in the array to change them
    for (dataset.inputs) |vec| {
        for (vec, 0..) |*val, j| {
            val.* = (val.* - mins[j]) / (maxs[j] - mins[j]);
        }
    }

    // dump the contents and free allocator for next use
    dataset.allocator.free(mins);
    dataset.allocator.free(maxs);
}
