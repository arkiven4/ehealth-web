const express = require('express');
const tbcareController = require('../controllers/tbcare');
const { isAuth } = require('../middlewares/auth');
const router = express.Router();
const doctorController = require('../controllers/doctor');

router.get('/', tbcareController.getLandingPage);
router.get('/download', tbcareController.getDownloadPage);
router.get('/about', tbcareController.getAboutPage);
router.get('/login', tbcareController.getTbcareLoginPage);
router.get('/predict', isAuth, tbcareController.getPredict);
router.post('/predict', isAuth, tbcareController.postPredict);
router.get('/patient-history', isAuth, tbcareController.getPatientHistoryList);
router.get('/patient-history/:patientId', isAuth, tbcareController.getPatientHistoryDetail);
router.get("/history/:predictionId", isAuth, tbcareController.getPredictionDetail);
router.get('/edit-patient/:patientId', isAuth, doctorController.tbcare_get_edit_patient);
router.post('/edit-patient', isAuth, doctorController.tbcare_post_edit_patient);
router.post('/delete-patient', isAuth, doctorController.tbcare_delete_patient);

router.get('/getPredict_filteredWav', isAuth, tbcareController.getPredict_filteredWav);

module.exports = router;