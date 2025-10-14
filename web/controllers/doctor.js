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
      // If request contains tbcare-specific fields, create TbcareProfile and link
      try {
        const newUserId = user.upserted[0]._id;
        // check if tbcare fields present
        if (req.body.age || req.body.sex) {
          const newProfile = new TbcareProfile({
            user: newUserId,
            sex: req.body.sex || null,
            age: req.body.age || null,
            height: req.body.height || null,
            weight: req.body.weight || null,
            bmi: req.body.bmi || null,
            weightStatus: req.body.weightStatus || null,
            isCoughProductive: req.body.isCoughProductive || null,
            coughDurationDays: req.body.coughDurationDays || null,
            hasHemoptysis: req.body.hasHemoptysis || false,
            hasChestPain: req.body.hasChestPain || false,
            hasShortBreath: req.body.hasShortBreath || false,
            hasFever: req.body.hasFever || false,
            hasNightSweats: req.body.hasNightSweats || false,
            hasWeightLoss: req.body.hasWeightLoss || false,
            weightLossAmountKg: req.body.weightLossAmountKg || null,
            tobaccoUse: req.body.tobaccoUse || null,
            cigarettesPerDay: req.body.cigarettesPerDay || null,
            smokingSinceMonths: req.body.smokingSinceMonths || null,
            stoppedSmokingMonths: req.body.stoppedSmokingMonths || null,
            hadPriorTB: req.body.hadPriorTB || null,
            comorbidities: req.body.comorbidities || [],
            comorbiditiesOther: req.body.comorbiditiesOther || null,
          });
          const savedProfile = await newProfile.save();
          await User.updateOne({ _id: newUserId }, { $set: { tbcareProfile: savedProfile._id } });
        }
      } catch (e) {
        console.error('Failed creating tbcare profile during create_patient:', e.message);
      }
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
  res.render("doctor/tbcare/add-patient", {
    pageTitle: "TBCare - Tambah Pasien",
    pageHeader: "Add New TBCare Patient",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user
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

    // 2. Buat dokumen TbcareProfile yang terhubung dengan User
    const newProfile = new TbcareProfile({
      user: savedUser._id,
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
    req.flash('error', 'Failed to create patient. Email may already exist.');
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
      audioFolders: audioFolders,
      errorMessage: req.flash('error'),
      hasResult: false, 
      predictionResult: null,
      predictionDetail: null,
      waveform: null,
      mfcc: null,
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
/**
 * @description Menampilkan halaman untuk mengedit data pasien TBCare.
 * GET Request.
 */
exports.tbcare_get_edit_patient = async (req, res, next) => {
  try {
    const patientId = req.params.patientId;
    const patient = await User.findOne({ _id: patientId, doctor: req.session.user._id }).populate('tbcareProfile');

    if (!patient) {
      req.flash('error', 'Patient not found.');
      return res.redirect('/tbcare/patient-history');
    }

    res.render('doctor/tbcare/edit-patient', {
      pageTitle: 'Edit Patient',
      pageHeader: `Edit TBCare Patient: ${patient.fullName.first}`,
      userdata: req.session.user,
      patient: patient,
      csrfToken: req.csrfToken(),
      role: req.session.user.role,
      subrole: req.session.user.subrole
    });

  } catch (err) {
    console.log(err);
    next(err);
  }
};

/**
 * @description Memproses data dari form edit pasien TBCare dan menyimpannya ke database.
 */
exports.tbcare_post_edit_patient = async (req, res, next) => {
  const {
    patientId,
    fname,
    lname,
    email,
    uname,
    sex,
    age,
    height,
    weight,
    bmi,
    weightStatus,
    isCoughProductive,
    coughDurationDays,
    hasHemoptysis,
    hasChestPain,
    hasShortBreath,
    hasFever,
    hasNightSweats,
    hasWeightLoss,
    weightLossAmountKg,
    tobaccoUse,
    cigarettesPerDay,
    smokingSinceMonths,
    stoppedSmokingMonths,
    hadPriorTB,
    comorbidities,
    comorbiditiesOther
  } = req.body;

  try {
    await User.updateOne({ _id: patientId, doctor: req.session.user._id }, {
      $set: {
        'fullName.first': fname,
        'fullName.last': lname,
        email: email,
        userName: uname
      }
    });

    await TbcareProfile.updateOne({ user: patientId }, {
      $set: {
        sex: sex,
        age: age,
        height: height,
        weight: weight,
        bmi: bmi,
        weightStatus: weightStatus,
        isCoughProductive: isCoughProductive,
        coughDurationDays: coughDurationDays,
        hasHemoptysis: hasHemoptysis,
        hasChestPain: hasChestPain,
        hasShortBreath: hasShortBreath,
        hasFever: hasFever,
        hasNightSweats: hasNightSweats,
        hasWeightLoss: hasWeightLoss,
        weightLossAmountKg: weightLossAmountKg,
        tobaccoUse: tobaccoUse,
        cigarettesPerDay: cigarettesPerDay,
        smokingSinceMonths: smokingSinceMonths,
        stoppedSmokingMonths: stoppedSmokingMonths,
        hadPriorTB: hadPriorTB,
        comorbidities: comorbidities,
        comorbiditiesOther: comorbiditiesOther
      }
    });

    req.flash('success', 'Patient data updated successfully.');
    res.redirect('/tbcare/patient-history');

  } catch (err) {
    console.log(err);
    req.flash('error', 'Failed to update patient data. Email or username might already be in use.');
    res.redirect(`/tbcare/edit-patient/${patientId}`);
  }
};

/**
 * @description Menghapus data pasien, profil, dan semua riwayat prediksinya
 */
exports.tbcare_delete_patient = async (req, res, next) => {
  const { patientId } = req.body;

  try {
    await Prediction.deleteMany({ patient: patientId });
    await TbcareProfile.deleteOne({ user: patientId });
    await User.deleteOne({ _id: patientId, doctor: req.session.user._id });
    await User.updateOne({ _id: req.session.user._id }, { $pull: { patient: patientId } });

    req.flash('success', 'Patient has been deleted successfully.');
    res.redirect('/tbcare/patient-history');
    
  } catch (err) {
    console.log(err);
    req.flash('error', 'Failed to delete patient.');
    res.redirect('/tbcare/patient-history');
  }
};

// pembatashhh
