const tf = require('@tensorflow/tfjs');
require("@tensorflow/tfjs-node");
const fs = require("fs");

exports.train = async (options, data_set) => {
  var defaultParams = {
    epochs: 100,
    train_size: 70,
    learn_rate: 0.01,
    hid_layers: 4,
    win_size: 50,
    input_layer_neurons: 100,
    rnn_input_layer_features: 10,
    rnn_output_neurons: 20,
    output_layer_neurons: 1
  };

  //merge options
  var params = {
    ...defaultParams,
    ...options,
  };

  //calculate additional parameters
  params.input_layer_shape = params.win_size;
  params.rnn_input_layer_timesteps = params.input_layer_neurons / params.rnn_input_layer_features;
  params.rnn_input_shape = [params.rnn_input_layer_features, params.rnn_input_layer_timesteps];
  params.rnn_batch_size = params.win_size;
  params.output_layer_shape = params.rnn_output_neurons;

  //train
  const sma_vec = ComputeSMA(data_set, params.win_size);
  const io = ComputeIO(sma_vec);

  const model = tf.sequential();

  let X = io.inputs.slice(0, Math.floor(params.train_size / 100 * io.inputs.length));
  let Y = io.outputs.slice(0, Math.floor(params.train_size / 100 * io.outputs.length));

  const xs = tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10));
  const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]).div(tf.scalar(10));

  model.add(tf.layers.dense({
    units: params.input_layer_neurons,
    inputShape: [params.input_layer_shape]
  }));
  model.add(tf.layers.reshape({
    targetShape: params.rnn_input_shape
  }));

  var lstm_cells = [];
  for (var index = 0; index < params.hid_layers; index++) {
    lstm_cells.push(tf.layers.lstmCell({
      units: params.rnn_output_neurons
    }));
  }

  model.add(tf.layers.rnn({
    cell: lstm_cells,
    inputShape: params.rnn_input_shape,
    returnSequences: false
  }));

  model.add(tf.layers.dense({
    units: params.output_layer_neurons,
    inputShape: [params.output_layer_shape]
  }));

  model.compile({
    optimizer: tf.train.adam(params.learn_rate),
    loss: 'meanSquaredError'
  });

  var hist = await model.fit(xs, ys, {
    batchSize: params.rnn_batch_size,
    epochs: params.epochs,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        console.log(epoch);
        console.log(log);
      }
    }
  });

  await model.save('file://model/');

  //Simple Moving Average
  function ComputeSMA(data, win_size) {
    var avgs = [];
    for (var i = 0; i <= data.length - params.win_size; i++) {
      var curr = 0.00,
        t = i + win_size;
      for (var k = i; k < t && k <= data.length; k++) {
        curr += data[k].price / win_size;
      }
      avgs.push({
        set: data.slice(i, i + win_size),
        avg: curr
      });
    }
    return avgs;
  }

  function ComputeIO(vec) {
    var inputs = vec.map(function (inp) {
      return inp.set.map(function (val) {
        return parseFloat(val.price);
      });
    });
    var outputs = vec.map(function (outp) {
      return outp.avg;
    });

    return {
      inputs: inputs,
      outputs: outputs
    };
  }
};