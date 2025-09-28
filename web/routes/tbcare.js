const express = require('express');
const tbcareController = require('../controllers/tbcare');

const router = express.Router();
router.get('/', tbcareController.getLandingPage);
router.get('/download', tbcareController.getDownloadPage);
router.get('/about', tbcareController.getAboutPage);
router.get('/login', tbcareController.getTbcareLoginPage);
router.get('/predict', isAuth, tbcareController.getPredict);
router.post('/predict', isAuth, upload.single('audiofile'), tbcareController.postPredict);
module.exports = router;