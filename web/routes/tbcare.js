const express = require('express');
const tbcareController = require('../controllers/tbcare');

const router = express.Router();
router.get('/', tbcareController.getLandingPage);
module.exports = router;