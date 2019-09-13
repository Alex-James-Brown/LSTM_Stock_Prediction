const {
  train
} = require("./train");

const axios = require("axios");

axios.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo")
  .then((response) => {
    console.log(response);
    var days = response.data["Time Series (Daily)"];
    var data_set = [];
    for (let date in days) {
      data_set.push({
        timestamp: date,
        price: parseFloat(days[date]["4. close"])
      });
    }
    data_set.reverse();
    train({}, data_set);
  });