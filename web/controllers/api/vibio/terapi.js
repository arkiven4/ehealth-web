const Vibio_terapi = require("../../../models/vibio_terapi");
const initParam = require("../../../helpers/init");

exports.terapiData = async (req, res, next) => {
  var uniqueID = new Date().getTime().toString(36);
  if (Object.keys(req.body).length != 0) {
    try {
      const terapi = await Vibio_terapi.create({ uuid: uniqueID, uuid_user: req.params.uuid_user, json_data: req.body.json_data, terapi: req.body.tipe_terapi });
      if (terapi) {
        res.json({
          status: "success",
          code: 200,
          message: "Success Insert Data",
        });
      } else {
        res.json({
          status: "error",
          code: 404,
          message: terapi,
        });
      }
    } catch (error) {
      res.json({
        status: "error",
        code: 404,
        message: terapi,
      });
    }
  } else {
    res.json({
      status: "error",
      code: 404,
      message: "Empty Data",
    });
  }
};
