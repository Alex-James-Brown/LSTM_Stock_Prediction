exports.SMA = (data, win_size) => {
    var avgs = [];
    for (var i = 0; i <= data.length - win_size; i++) {
        var curr = 0.0,
        t = i + win_size;
        for (var k = i; k < t && k <= data.length; k++) {
            curr += data[k].price / win_size;
        }
        avgs.push({
            timestamp: data[i].timestamp,
            set: data.slice(i, i + win_size),
            avg: curr
        });
    }
    return avgs;
}