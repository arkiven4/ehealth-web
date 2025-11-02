const mongoose = require('mongoose');
const session = require('express-session');
const MongoDBStore = require('connect-mongodb-session')(session);

const initParam = require('../helpers/init');

const MONGODB_URI = initParam.MONGODB_URI;
const Device_Data_Cough_TBPrimer = require("../models/device_data_cough_tbprimer.js");

(async () => {
  await mongoose.connect(MONGODB_URI);

  const docs = await Device_Data_Cough_TBPrimer.find({}).lean();
  console.log("Found", docs.length, "docs");

  for (const d of docs) {
    try {
      const parsed = JSON.parse(d.json_data);
      await Device_Data_Cough_TBPrimer.updateOne(
        { _id: d._id },
        { $set: { json_data: parsed } }
      );
    } catch (e) {
      console.error("Parse error for", d._id, e.message);
    }
  }

  console.log("Migration done");
  process.exit(0);
})();