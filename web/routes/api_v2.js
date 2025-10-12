const express = require('express');
const router = express.Router();

const apiControllerV2 = require('../controllers/api_v2');

router.post(
    '/start-prediction', 
    apiControllerV2.postStartPrediction
);

router.post(
    '/create-test-patient',
    apiControllerV2.postCreateTestPatient
);

router.get(
    '/patient-history/:participantId',
    apiControllerV2.getPatientHistory
);

router.put(
    '/prediction/:predictionId',
    apiControllerV2.updatePrediction
);

router.delete(
    '/prediction/:predictionId',
    apiControllerV2.deletePrediction
);

module.exports = router;