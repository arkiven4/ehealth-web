const User = require("../models/user");
const fs = require('fs');
const path = require('path');

const bcrypt = require("bcryptjs");
const { spawn } = require('child_process');
const path = require('path');

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
  const { patientId, coughFilePath } = req.body;

  if (!patientId || !coughFilePath) {
    req.flash('error', 'Please select a patient and a cough file.');
    return res.redirect('/sub_1/predict');
  }

  const coughFileAbsolutePath = path.join(__dirname, '..', coughFilePath);
  const pythonScriptPath = path.join(__dirname, '..', 'python-script', 'process_cough.py');

  const pythonProcess = spawn('python3', [pythonScriptPath, coughFileAbsolutePath]);

  let predictionResult = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    predictionResult += data.toString().trim();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  pythonProcess.on('close', async (code) => {
    if (code !== 0 || errorOutput) {
      console.error(`Python Error: ${errorOutput}`);
      req.flash('error', 'Prediction failed. Please check server logs.');
      return res.redirect('/sub_1/predict');
    }

    console.log('Raw Prediction Result:', predictionResult);
    
    // Parse hasilnya: "KEPUTUSAN_AKHIR,JUMLAH_TB,JUMLAH_NON_TB,TOTAL_SEGMEN_BATUK"
    const parts = predictionResult.split(',');
    
    if(parts.length < 4){
        // Jika output bukan format yang diharapkan, anggap sebagai pesan error
        req.flash('info', `Prediction Info: ${predictionResult}`);
        return res.redirect('/sub_1/predict');
    }

    const [finalDecision, tbSegments, nonTbSegments, totalCoughSegments] = parts;

    // --- Simpan hasil ke database ---
    // Anda perlu membuat model baru untuk menyimpan data ini, misal `Prediction.js`
    // try {
    //   await Prediction.create({
    //     patient: patientId,
    //     audioFile: coughFilePath,
    //     result: finalDecision,
    //     tbSegmentCount: parseInt(tbSegments),
    //     nonTbSegmentCount: parseInt(nonTbSegments),
    //     totalCoughSegments: parseInt(totalCoughSegments),
    //     predictedBy: req.session.user._id
    //   });
    // } catch (dbError) {
    //    console.error('Database Error:', dbError);
    //    req.flash('error', 'Failed to save prediction result.');
    //    return res.redirect('/sub_1/predict');
    // }

    // Tampilkan hasil menggunakan flash message
    req.flash('success', `Prediction Complete! Result: ${finalDecision} (TB Segments: ${tbSegments}, Non-TB: ${nonTbSegments})`);
    res.redirect("/sub_1/doctor"); // Arahkan ke dashboard dokter
  });
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
