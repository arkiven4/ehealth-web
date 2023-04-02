const express = require("express");
const multer  = require('multer')
const upload = multer({ dest: './temp/' })

const apiController = require("../controllers/api");
const apiControllerDevice = require("../controllers/api/device/device");
const apiTerapiController = require("../controllers/api/vibio/terapi");
const isAuthApi = require('../middlewares/is-aut-user-api');

const router = express.Router();

//Auth API
router.post('/api/admin/login', apiController.login);

//Admin Api
router.post('/api/admin/home', isAuthApi ,apiController.home);
router.post('/api/admin/data-batuk', isAuthApi ,apiController.data_batuk);
router.post('/api/admin/device-list', isAuthApi ,apiController.device_list);
router.post('/api/admin/device-detail', isAuthApi ,apiController.device_detail);

//Device API
router.post('/api/device/sendData/:device_id', upload.any() , apiControllerDevice.sendData);
router.get('/api/device/testAPI', apiControllerDevice.testAPI);

// Vibio
router.post('/api/vibio/insert_terapi/:uuid_user', upload.any() , apiTerapiController.terapiData);

//General Api
router.post('/api/submit-data-batuk', isAuthApi ,  upload.any() , apiController.submit_data_batuk);

module.exports = router;
