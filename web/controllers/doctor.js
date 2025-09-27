const User = require("../models/user");
const fs = require('fs');
const path = require('path');

const bcrypt = require("bcryptjs");

exports.home = async (req, res, next) => {
  const pasient = await User.find({ role: "patient" , doctor : req.session.user._id});
  res.render("doctor/home-doctor", {
    pageTitle: "E-Health Dashboard",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    pasient : pasient
  });
};

exports.add_patient = async (req, res, next) => {
  res.render("doctor/add-patient", {
    pageTitle: "E-Health Dashboard",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
  });
};

// TBCare (PKM)
exports.tbcare_home = async (req, res, next) => {
  const pasien = await User.find({ role: "patient", doctor: req.session.user._id });
  res.render("doctor/tbcare/home-doctor", {
    pageTitle: "TBCare Dashboard",
    pageHeader: "TBCare Dashboard",
    isTBCare: true,
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    pasient: pasien,
    userdata: req.session.user,
  });
};

exports.tbcare_predict_form = async (req, res, next) => {
  const patients = await User.find({ role: "patient", doctor: req.session.user._id });
  
  res.render("doctor/tbcare/predict", {
    pageTitle: "TBCare - Cough Prediction",
    pageHeader: "Cough Recording Prediction",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
    patients: patients
  });
};

exports.tbcare_predict = async (req, res, next) => {
  const pasien = await User.find({ role: "patient", doctor: req.session.user._id });
  res.render("doctor/tbcare/predict", { 
    pageTitle: "TBCare - Predict",
    pageHeader: "Predict Cough Analysis",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
  });
};

exports.tbcare_predict_form = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id });

    const tbcareUploadsPath = path.join(__dirname, '..', 'public', 'uploads', 'tbcare');
    let coughFiles = [];

    if (fs.existsSync(tbcareUploadsPath)) {
      coughFiles = fs.readdirSync(tbcareUploadsPath)
        .filter(file => file.endsWith('.wav'))
        .map(file => {
          const stats = fs.statSync(path.join(tbcareUploadsPath, file));
          return {
            name: file,
            modifiedTime: stats.mtime.getTime(), 
            displayDate: stats.mtime.toISOString().split('T')[0] 
          };
        })
        .sort((a, b) => b.modifiedTime - a.modifiedTime);
    }

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction",
      pageHeader: "Cough Recording Prediction",
      role: req.session.user.role,
      subrole: req.session.user.subrole,
      userdata: req.session.user,
      patients: patients,
      coughFiles: coughFiles
    });
  } catch (error) {
    console.log(error);
    next(error);
  }
};

exports.tbcare_add_patient = async (req, res, next) => {
  const lastPatient = await User.findOne({ 'tbcareProfile.participantId': { $exists: true } })
                                  .sort({ createdAt: -1 });

  let nextIdNumber = 1;
  if (lastPatient && lastPatient.tbcareProfile.participantId) {
    const lastId = lastPatient.tbcareProfile.participantId;
    const lastNumber = parseInt(lastId.replace('300P', ''), 10);
    nextIdNumber = lastNumber + 1;
  }

  const nextParticipantId = '300P' + String(nextIdNumber).padStart(5, '0');

  res.render("doctor/tbcare/add-patient", {
    pageTitle: "TBCare - Tambah Pasien",
    pageHeader: "Add New TBCare Patient",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
    nextParticipantId: nextParticipantId // <-- Kirim ID baru ke view
  });
};

exports.tbcare_create_patient = async (req, res, next) => {
  if (req.body.pass === req.body.rpass) {
    const hashedPw = await bcrypt.hash(req.body.pass, 12);
    
    await User.create({
      email: req.body.email,
      userName: req.body.uname,
      password: hashedPw,
      fullName: { first: req.body.fname, last: req.body.lname },
      city: req.body.city,
      mobileNumber1: req.body.mobno,
      address1: req.body.add1,
      role: "patient",
      doctor: req.session.user._id,

      tbcareProfile: {
        participantId: req.body.participantId,
        sex: req.body.sex,
        age: req.body.age,
        height: req.body.height,
        weight: req.body.weight,
        coughDuration: req.body.coughDuration,
        isCoughProductive: req.body.isCoughProductive,
        hasHemoptysis: req.body.hasHemoptysis,
        hasChestPain: req.body.hasChestPain,
        hasShortBreath: req.body.hasShortBreath,
        hasFever: req.body.hasFever,
        hasNightSweats: req.body.hasNightSweats,
        hasWeightLoss: req.body.hasWeightLoss,
        tobaccoUse: req.body.tobaccoUse,
        cigarettesPerDay: req.body.cigarettesPerDay,
        hadPriorTB: req.body.hadPriorTB,
      }
    });
  }
  res.redirect("/sub_1/doctor");
};

exports.tbcare_predict_post = async (req, res, next) => {
  const patientId = req.body.patientId;
  const coughFilePath = req.body.coughFilePath;

  console.log("Patient ID:", patientId);
  console.log("Cough File Path:", coughFilePath);

  if (!coughFilePath) {
    return res.redirect('/sub_1/predict');
  }

  // need ml logic here

  // temp
  res.redirect("/sub_1/doctor");
};

// pembatashhh

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
