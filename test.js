process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

require("dotenv").config();
var fs = require('fs');
const { SMA } = require("./functions/simple-moving-average");
const { ComputeIO } = require("./functions/compute-io");

var trainer = require("./train-brain")({});
const axios = require("axios");

var plotly = require("plotly")(process.env.plotlyUsername, process.env.plotlyApiKey);

var data_set = [];
axios
  .get(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo"
  )
  .then(async response => {
    var days = response.data["Time Series (Daily)"];
    for (let date in days) {
      data_set.push({
        timestamp: date,
        price: parseFloat(days[date]["4. close"])
      });
    }
    data_set.reverse();

    var sma_vec = SMA(data_set, 50);
    var data = ComputeIO(sma_vec, "price", "avg");
    await trainer.trainNetwork(data);
    trainer.saveNetwork("./model", "model.json");

    var trainingData = {
      type: "scatter",
      x: data_set.map(item => {
        return item.timestamp;
      }),
      y: data_set.map(item => {
        return item.price;
      })
    }

    var smaData = {
      type: "scatter",
      x: sma_vec.map(item => {
        return item.timestamp;
      }),
      y: sma_vec.map(item => {
        return item.avg;
      })
    }

    var chartData = [
      trainingData,
      smaData
    ];

    var figure = {
      data: chartData
    };

    var graphOptions = {
      format: 'png',
      width: 1000,
      height: 500
    };

    plotly.getImage(figure, graphOptions, function (error, stream) {
      if (error) return console.log(error);

      var fileStream = fs.createWriteStream('chart.png');
      stream.pipe(fileStream);
    });
  });