const User = require("../models/user");
const fs = require('fs');
const path = require('path');

const bcrypt = require("bcryptjs");
const { spawn } = require('child_process');
const TbcareProfile = require('../models/tbcare_profile');
const Prediction = require('../models/tbcare_prediction');

exports.home = async (req, res, next) => {
  const pasient = await User.find({ role: "patient" , doctor : req.session.user._id});
  res.render("doctor/home-doctor", {
    pageTitle: "E-Health Dashboard",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    pasient : pasient,
    userdata: req.session.user,
  });
};

exports.add_patient = async (req, res, next) => {
  res.render("doctor/add-patient", {
    pageTitle: "E-Health Dashboard",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
  });
};

exports.create_patient = async (req, res, next) => {
  if (req.body.pass === req.body.rpass) {
    const hashedPw = await bcrypt.hash(req.body.pass, 12);
    const user = await User.update(
      { email: req.body.email, userName: req.body.uname },
      {
        email: req.body.email,
        userName: req.body.uname,
        fullName: { first: req.body.fname, last: req.body.lname },
        city: req.body.city,
        mobileNumber1: req.body.mobno,
        mobileNumber2: req.body.altconno,
        address1: req.body.add1,
        address2: req.body.add2,
        country: req.body.selectcountry,
        pinCode: req.body.pno,
        password: hashedPw,
        role: "patient",
        doctor: req.session.user._id,
      },
      { upsert: true }
    );
    if (user.upserted.length > 0) {
      User.updateOne(
        { _id: req.session.user._id },
        { $push: { patient: user.upserted[0]._id } }
      ).then((result) => {});
    }
  }
  res.redirect("/add-patient");
};

// TBCare (PKM)
exports.tbcare_home = async (req, res, next) => {
  const pasien = await User.find({ role: "patient", doctor: req.session.user._id });
  res.render("doctor/tbcare/home-doctor", {
    pageTitle: "TBCare Dashboard",
    pageHeader: "TBCare Dashboard",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    pasient: pasien,
    userdata: req.session.user,
  });
};

exports.tbcare_add_patient = async (req, res, next) => {
  const lastPatient = await User.findOne({ 'tbcareProfile.participantId': { $exists: true } }).sort({ createdAt: -1 });
  let nextIdNumber = 1;
  if (lastPatient && lastPatient.tbcareProfile.participantId) {
    const lastNumber = parseInt(lastPatient.tbcareProfile.participantId.replace('300P', ''), 10);
    nextIdNumber = lastNumber + 1;
  }
  const nextParticipantId = '300P' + String(nextIdNumber).padStart(5, '0');
  res.render("doctor/tbcare/add-patient", {
    pageTitle: "TBCare - Tambah Pasien",
    pageHeader: "Add New TBCare Patient",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
    nextParticipantId: nextParticipantId
  });
};

exports.tbcare_create_patient = async (req, res, next) => {
  if (req.body.pass !== req.body.rpass) {
    req.flash('error', 'Passwords do not match.');
    return res.redirect('/sub_1/add-patient');
  }

  try {
    const hashedPw = await bcrypt.hash(req.body.pass, 12);

    const newUser = new User({
      email: req.body.email,
      userName: req.body.uname,
      password: hashedPw,
      fullName: { first: req.body.fname, last: req.body.lname },
      city: req.body.city,
      mobileNumber1: req.body.mobno,
      address1: req.body.add1,
      role: "patient",
      accountType: 'tbcare', 
      doctor: req.session.user._id,
    });
    
    const savedUser = await newUser.save();

    const newProfile = new TbcareProfile({
      user: savedUser._id,
      participantId: req.body.participantId,
      sex: req.body.sex,
      age: req.body.age,
      height: req.body.height,
      weight: req.body.weight,
      bmi: req.body.bmi,
      weightStatus: req.body.weightStatus,
      isCoughProductive: req.body.isCoughProductive,
      coughDurationDays: req.body.coughDurationDays,
      hasHemoptysis: req.body.hasHemoptysis,
      hasChestPain: req.body.hasChestPain,
      hasShortBreath: req.body.hasShortBreath,
      hasFever: req.body.hasFever,
      hasNightSweats: req.body.hasNightSweats,
      hasWeightLoss: req.body.hasWeightLoss,
      weightLossAmountKg: req.body.weightLossAmountKg,
      tobaccoUse: req.body.tobaccoUse,
      cigarettesPerDay: req.body.cigarettesPerDay,
      smokingSinceMonths: req.body.smokingSinceMonths,
      stoppedSmokingMonths: req.body.stoppedSmokingMonths,
      hadPriorTB: req.body.hadPriorTB,
      comorbidities: req.body.comorbidities,
      comorbiditiesOther: req.body.comorbiditiesOther,
    });

    const savedProfile = await newProfile.save();

    savedUser.tbcareProfile = savedProfile._id;
    await savedUser.save();

    res.redirect("/sub_1/doctor");

  } catch (err) {
    console.log(err);
    req.flash('error', 'Failed to create patient. Email or Participant ID may already exist.');
    res.redirect('/sub_1/add-patient');
  }
};

exports.tbcare_getEditPatient = async (req, res, next) => {
    try {
        const patientId = req.params.patientId;
        const patient = await User.findById(patientId).populate('tbcareProfile');
        if (!patient || patient.doctor.toString() !== req.session.user._id.toString()) {
            req.flash('error', 'Patient not found or you do not have permission to edit.');
            return res.redirect('/tbcare/patient-history');
        }

        res.render('doctor/tbcare/edit-patient', {
            pageTitle: 'Edit Patient',
            pageHeader: `Edit Patient: ${patient.fullName.first}`,
            userdata: req.session.user,
            patient: patient,
            csrfToken: req.csrfToken(),
            errorMessage: req.flash('error')[0]
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @description Memproses form update data pasien TBCare.
 */
exports.tbcare_postUpdatePatient = async (req, res, next) => {
    try {
        const { 
            patientId,
            fname, lname, email, mobno, add1, city,
            age, sex, height, weight, bmi, weightStatus,
            isCoughProductive, coughDurationDays, hasWeightLoss, weightLossAmountKg,
            hasHemoptysis, hasChestPain, hasShortBreath, hasFever, hasNightSweats,
            hadPriorTB, tobaccoUse, cigarettesPerDay, smokingSinceMonths, stoppedSmokingMonths,
            comorbidities, comorbiditiesOther
        } = req.body;

        const user = await User.findById(patientId);
        if (!user) {
            req.flash('error', 'Patient not found.');
            return res.redirect('/tbcare/patient-history');
        }

        // Update User model
        user.fullName.first = fname;
        user.fullName.last = lname;
        user.email = email;
        user.mobileNumber1 = mobno;
        user.address1 = add1;
        user.city = city;
        await user.save();

        // Update TbcareProfile model
        const profile = await TbcareProfile.findOne({ user: patientId });
        if (profile) {
            profile.age = age;
            profile.sex = sex;
            profile.height = height;
            profile.weight = weight;
            profile.bmi = bmi;
            profile.weightStatus = weightStatus;
            profile.isCoughProductive = isCoughProductive;
            profile.coughDurationDays = coughDurationDays;
            profile.hasWeightLoss = hasWeightLoss;
            profile.weightLossAmountKg = weightLossAmountKg;
            profile.hasHemoptysis = hasHemoptysis;
            profile.hasChestPain = hasChestPain;
            profile.hasShortBreath = hasShortBreath;
            profile.hasFever = hasFever;
            profile.hasNightSweats = hasNightSweats;
            profile.hadPriorTB = hadPriorTB;
            profile.tobaccoUse = tobaccoUse;
            profile.cigarettesPerDay = cigarettesPerDay;
            profile.smokingSinceMonths = smokingSinceMonths;
            profile.stoppedSmokingMonths = stoppedSmokingMonths;
            profile.comorbidities = comorbidities;
            profile.comorbiditiesOther = comorbiditiesOther;
            await profile.save();
        }

        res.redirect('/tbcare/patient-history');
    } catch (error) {
        console.log("Error updating patient:", error);
        req.flash('error', 'Failed to update patient data.');
        res.redirect('/tbcare/patient-history');
    }
};

/**
 * @description Menghapus pasien TBCare dan semua data terkaitnya
 */
exports.tbcare_deletePatient = async (req, res, next) => {
    try {
        const { patientId } = req.params;

        const patient = await User.findById(patientId);
        if (!patient || patient.doctor.toString() !== req.session.user._id.toString()) {
            return res.status(403).json({ message: 'Forbidden: You do not have permission to delete this patient.' });
        }

        await TbcarePrediction.deleteMany({ patient: patientId });
        await TbcareProfile.deleteOne({ user: patientId });
        const result = await User.findByIdAndDelete(patientId);

        if (!result) {
            return res.status(404).json({ message: 'Gagal menghapus: Pasien tidak ditemukan.' });
        }

        res.status(200).json({ message: 'Pasien dan semua data riwayatnya berhasil dihapus.' });
    } catch (error) {
        console.error("Error deleting patient:", error);
        res.status(500).json({ message: 'Gagal menghapus pasien.', details: error.message });
    }
};

// pembatashhh
