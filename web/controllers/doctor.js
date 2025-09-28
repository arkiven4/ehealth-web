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
    // Handle jika password tidak cocok
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

    // 2. Buat dokumen TbcareProfile yang terhubung dengan User
    const newProfile = new TbcareProfile({
      user: savedUser._id, // Hubungkan dengan ID user yang baru dibuat
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

exports.tbcare_predict_form = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id });
    const uploadsPath = path.join(__dirname, '..', 'public', 'uploads', 'tbcare');

    let audioFolders = [];
    let allFiles = [];

    const mainFolders = fs.readdirSync(uploadsPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);

    mainFolders.forEach(folder => {
        const folderPath = path.join(uploadsPath, folder);
        const files = fs.readdirSync(folderPath)
            .filter(file => file.endsWith('.wav'))
            .map(file => {
                const stats = fs.statSync(path.join(folderPath, file));
                return {
                    name: file,
                    folder: folder, 
                    path: `public/uploads/tbcare/${folder}/${file}`,
                    modifiedTime: stats.mtime.getTime(),
                    displayDate: stats.mtime.toISOString().split('T')[0]
                };
            });
        if(files.length > 0) {
            audioFolders.push(folder);
            allFiles.push(...files);
        }
    });

    allFiles.sort((a, b) => b.modifiedTime - a.modifiedTime);

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction",
      pageHeader: "Cough Recording Prediction",
      role: req.session.user.role,
      subrole: req.session.user.subrole,
      userdata: req.session.user,
      patients: patients,
      coughFiles: allFiles,
      audioFolders: audioFolders
    });
  } catch (error) {
    console.log(error);
    next(error);
  }
};

exports.tbcare_predict_post = async (req, res, next) => {
  const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

  if (!patientId || !coughFilePath || !sputumStatus) {
    req.flash('error', 'Please complete all required fields.');
    return res.redirect('/sub_1/predict');
  }

  let sputumConditionLabel = sputumStatus;
  if (sputumStatus === 'Sputum +' && sputumLevel) {
    sputumConditionLabel += ` (${sputumLevel})`;
  }

  const coughFileAbsolutePath = path.join(__dirname, '..', coughFilePath);
  const pythonScriptPath = path.join(__dirname, '..', 'python-script', 'process_cough.py');
  const pythonProcess = spawn('python3', [pythonScriptPath, coughFileAbsolutePath]);

  let predictionResult = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => { predictionResult += data.toString().trim(); });
  pythonProcess.stderr.on('data', (data) => { errorOutput += data.toString(); });

  pythonProcess.on('close', async (code) => {
    if (code !== 0 || errorOutput) {
      console.error(`Python Error: ${errorOutput}`);
      req.flash('error', 'Prediction failed. Please check server logs.');
      return res.redirect('/sub_1/predict');
    }

    const parts = predictionResult.split(',');
    if (parts.length < 4) {
      req.flash('info', `Prediction Info: ${predictionResult}`);
      return res.redirect('/sub_1/predict');
    }

    const [finalDecision, tbSegments, nonTbSegments, totalCoughSegments] = parts;

    try {
      await Prediction.create({
        patient: patientId,
        predictedBy: req.session.user._id,
        audioFile: coughFilePath,
        sputumCondition: sputumConditionLabel,
        result: finalDecision,
        tbSegmentCount: parseInt(tbSegments),
        nonTbSegmentCount: parseInt(nonTbSegments),
        totalCoughSegments: parseInt(totalCoughSegments)
      });

      req.flash('success', `Prediction Saved! Result: ${finalDecision}`);
      res.redirect("/sub_1/doctor");

    } catch (dbError) {
      console.error('Database Error:', dbError);
      req.flash('error', 'Failed to save prediction result.');
      return res.redirect('/sub_1/predict');
    }
  });
};

// pembatashhh
