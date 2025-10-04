const express = require('express');
const tbcareController = require('../controllers/tbcare');
const { isAuth } = require('../middlewares/auth');
const router = express.Router();

router.get('/', tbcareController.getLandingPage);
router.get('/download', tbcareController.getDownloadPage);
router.get('/about', tbcareController.getAboutPage);
router.get('/login', tbcareController.getTbcareLoginPage);
router.get('/predict', isAuth, tbcareController.getPredict);
router.post('/predict', isAuth, tbcareController.postPredict);
router.post('/save-prediction', isAuth, tbcareController.savePrediction);

module.exports = router;