const express = require('express');
const tbcareController = require('../controllers/tbcare');

const router = express.Router();
router.get('/', tbcareController.getLandingPage);
router.get('/download', tbcareController.getDownloadPage);
router.get('/about', tbcareController.getAboutPage);

module.exports = router;