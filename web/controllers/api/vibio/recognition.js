const Vibio_terapi = require("../../../models/vibio_terapi");
const Settings = require("../../../models/settings");
const initParam = require("../../../helpers/init");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

exports.setRecognitionServer = async (req, res, next) => {
  const update_setting = await Settings.updateOne(
    { key: req.body.key_setting },
    {
      value: req.body.value_setting,
    },
    { upsert: true }
  );

  if (update_setting.ok > 0) {
    res.json({
      status: "success",
      code: 200,
      message: "Success Insert/Update Data",
    });
  } else {
    res.json({
      status: "error",
      code: 404,
      message: "Failure Insert/Update Data",
    });
  }
};

exports.checkRecognitionServer = async (req, res, next) => {
  const recog_server = await Settings.findOne({ key: "vibio_recognition_server" });
  if (recog_server.value == '') {
    console.log(error);
    res.status(500).json({ error: "Internal Server Error" });
    return;
  }

  try {
    const response = await axios.get(recog_server.value, {
      headers: {
        "Ngrok-Skip-Browser-Warning": "true",
      },
    });

    const response_ngrok = response.data;

    res.json(response_ngrok);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
};

exports.recognitionSoundData = async (req, res, next) => {
  const recog_server = await Settings.findOne({ key: "vibio_recognition_server" });
  if (!recog_server) {
    console.log(error);
    res.status(500).json({ error: "Internal Server Error" });
    return;
  }

  var start_time = Date.now();
  try {
    const formData = new FormData();
    formData.append("file_audio", fs.createReadStream(req.files[0].path));

    const response = await axios.post(recog_server.value + "/recognition", formData, {
      headers: {
        ...formData.getHeaders(),
        "Ngrok-Skip-Browser-Warning": "true",
      },
    });

    const time_exec_local = ((Date.now() - start_time) / 1000).toFixed(2);
    const response_ngrok = response.data;
    response_ngrok.time_exec_local = time_exec_local;

    res.json(response_ngrok);
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Internal Server Error" });
  } finally {
    fs.unlinkSync(req.files[0].path);
  }
};