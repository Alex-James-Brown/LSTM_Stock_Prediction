const fs = require('fs');
const brain = require('brain.js');

module.exports = (user_config) => {
    return {
        trainNetwork: async (data_set) => {
            this.net = new brain.recurrent.LSTM({   
                inputSize: 50,
                hiddenLayers: [20],
                outputSize: 50
            });
            this.net.train(data_set, {
                iterations: 10,
                learningRate: 0.01,
                momentum: 0.1,
                errorThresh: 0.005,
                log: true,
                logPeriod: 1
            });
            return this.net;
        },
        saveNetwork: (dir, fname) => {
            if(!fs.existsSync(dir)) {
                fs.mkdirSync(dir);
            }
            fs.writeFile(`${dir}/${fname}`, JSON.stringify(this.net.toJSON()), "utf8", err => {
                if(err) console.log(err);
                console.log("saved");
            })
        }
    }
}



