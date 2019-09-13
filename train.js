const tf = require('@tensorflow/tfjs');
const fs = require("fs");

//www.alphavantage.co variables
const apikey = "demo";
const ticker = "MSFT";

//training data set
const win_size = 50;

//RNN LSTM hyper-params
const train_size = 70; //70% training, 30% testing
const epochs = 100;
const learn_rate = 0.01;
const hid_layers = 4;

//network config
const input_layer_shape  = win_size;
const input_layer_neurons = 100;

const rnn_input_layer_features = 10;
const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

const rnn_input_shape  = [rnn_input_layer_features, rnn_input_layer_timesteps];
const rnn_output_neurons = 20;

const rnn_batch_size = win_size;

const output_layer_shape = rnn_output_neurons;
const output_layer_neurons = 1;

//get data set then train network
fs.readFile('data.json', (err, data) => {
    var days = JSON.parse(data)["Time Series (Daily)"];

    var data_set = [];
    for(let date in days){
        data_set.push({ 
            timestamp: date, 
            price: parseFloat(days[date]["4. close"]) 
        });
    }

    data_set.reverse();
    train(data_set);
});
    
//Simple Moving Average
function ComputeSMA(data, win_size)
{
  var avgs = [];
  for (var i = 0; i <= data.length - win_size; i++){
    var curr = 0.00, t = i + win_size;
    for (var k = i; k < t && k <= data.length; k++){
      curr += data[k]['price'] / win_size;
    }
    avgs.push({ set: data.slice(i, i + win_size), avg: curr });
  }
  return avgs;
}

function ComputeIO(vec) {
    const inputs = vec.map(function(inp){ return inp['set'].map(function(val) { return parseFloat(val['price']); })});
    const outputs = vec.map(function(outp) { return outp['avg']; });

    return {
        inputs: inputs,
        outputs: outputs
    };
}

async function train(data_set){
    const sma_vec = ComputeSMA(data_set, win_size);
    const io = ComputeIO(sma_vec);

    const model = tf.sequential();

    let X = io.inputs.slice(0, Math.floor(train_size / 100 * io.inputs.length));
    let Y = io.outputs.slice(0, Math.floor(train_size / 100 * io.outputs.length));
  
    const xs = tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10));
    const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]).div(tf.scalar(10));
  
    model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
    model.add(tf.layers.reshape({targetShape: rnn_input_shape}));
  
    let lstm_cells = [];
    for (let index = 0; index < hid_layers; index++) {
         lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
    }
  
    model.add(tf.layers.rnn({
      cell: lstm_cells,
      inputShape: rnn_input_shape,
      returnSequences: false
    }));
  
    model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));
  
    model.compile({
      optimizer: tf.train.adam(learn_rate),
      loss: 'meanSquaredError'
    });
  
    const hist = await model.fit(xs, ys,
      { batchSize: rnn_batch_size, epochs: epochs, callbacks: {
        onEpochEnd: async (epoch, log) => {
          console.log(epoch);
          console.log(log);
        }
      }
    });

    await model.save('file://model/')
}