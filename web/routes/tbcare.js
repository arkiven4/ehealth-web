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
router.get('/patient-history', isAuth, tbcareController.getPatientHistoryList);
router.get('/patient-history/:patientId', isAuth, tbcareController.getPatientHistoryDetail);

module.exports = router;