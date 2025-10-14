const express = require("express");

const doctorController = require("../controllers/doctor");
const auth = require("../middlewares/auth");
const checkingRole = require("../middlewares/check-role");

const router = express.Router();

router.get(
  "/doctor",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.home
);

router.get(
  "/add-patient",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.add_patient
);

router.post(
  "/create-patient",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.create_patient
);

// TBCare (PKM)
router.get(
  "/sub_1/doctor",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.tbcare_home 
);

router.get(
  "/sub_1/add-patient",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.tbcare_add_patient 
);

router.post(
  "/sub_1/create-patient",
  auth.isAuth,
  checkingRole.isDoctor,
  doctorController.tbcare_create_patient 
);

router.get('/sub_1/edit-patient/:patientId', 
  auth.isAuth, 
  checkingRole.isDoctor,
  doctorController.tbcare_getEditPatient
);

router.post('/sub_1/update-patient', 
  auth.isAuth, 
  checkingRole.isDoctor,
  doctorController.tbcare_postUpdatePatient
);

router.delete(
    '/sub_1/tbcare/patient/:patientId',
    auth.isAuth,
    checkingRole.isDoctor,
    doctorController.tbcare_deletePatient
);

module.exports = router;
