exports.ComputeIO = (vec, input_name, output_name) => {
    var io = vec.map(function (item) {
        return {
            input: item.set.map(function (val) {
                return val[input_name];
            }), 
            output: item[output_name]
        }
    });
    return io;
}