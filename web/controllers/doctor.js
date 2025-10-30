const User = require("../models/user");
const fs = require('fs');
const path = require('path');
const bcrypt = require("bcryptjs");
const { spawn } = require('child_process');
const TbcareProfile = require('../models/tbcare_profile');
const Prediction = require('../models/tbcare_prediction');
const regions = require('../helpers/region');

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
            province: req.body.province || '',
            city: req.body.city || '',
            district: req.body.district || '',
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
function calculateAge(dateOfBirth) {
  if (!dateOfBirth) return null;
  const birthDate = new Date(dateOfBirth);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDifference = today.getMonth() - birthDate.getMonth();
  if (monthDifference < 0 || (monthDifference === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  return age;
}

exports.tbcare_home = async (req, res, next) => {
  try {
    const doctorId = req.session.user._id;
    const allPatients = await User.find({ 
      role: "patient", 
      accountType: 'tbcare', 
      doctor: doctorId 
    }).populate('tbcareProfile'); 
    const totalPatientCount = allPatients.length;

    const tbPredictions = await Prediction.find({ 
      predictedBy: doctorId, 
      result: { $regex: /TB/i } 
    }).distinct('patient'); 

    const tbCasesCount = tbPredictions.length;
    const surabayaPatients = await TbcareProfile.find({ 
      user: { $in: allPatients.map(p => p._id) }, // Hanya dari pasien dokter ini
      city: 'Surabaya' 
    }).select('district');
      
    const districtCounts = {};
    // Inisialisasi semua kecamatan Surabaya dengan 0
    regions.surabayaDistricts.forEach(district => {
        districtCounts[district] = 0; 
    });
    // Hitung pasien di setiap kecamatan
    surabayaPatients.forEach(p => {
        if(p.district && districtCounts.hasOwnProperty(p.district)) {
            districtCounts[p.district]++;
        }
    });

    res.render("doctor/tbcare/home-doctor", {
      pageTitle: "TBCare Dashboard",
      pageHeader: "TBCare Dashboard",
      role: req.session.user.role,
      subrole: req.session.user.subrole,
      pasien: allPatients,
      userdata: req.session.user,
      totalPatientCount: totalPatientCount,
      tbCasesCount: tbCasesCount,
      regionalStats: districtCounts,
      csrfToken: req.csrfToken()
    });
  } catch(err) {
      console.log(err);
      next(err);
  }
};

exports.tbcare_add_patient = async (req, res, next) => {
  res.render("doctor/tbcare/add-patient", {
    pageTitle: "TBCare - Tambah Pasien",
    pageHeader: "Add New TBCare Patient",
    role: req.session.user.role,
    subrole: req.session.user.subrole,
    userdata: req.session.user,
    csrfToken: req.csrfToken(),
    provinces: regions.provinces,
    surabayaDistricts: regions.surabayaDistricts
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

    // simpan/attach tbcare profile termasuk lokasi jika ada
    try {
      const profileData = {
        user: savedUser._id,
        participantId: req.body.participantId || '',
        province: req.body.province || '',
        city: req.body.city || '',
        district: req.body.district || '',
        dateOfBirth: req.body.dateOfBirth ? new Date(req.body.dateOfBirth) : null,
        height: req.body.height ? parseFloat(req.body.height) : null,
        weight: req.body.weight ? parseFloat(req.body.weight) : null,
        bmi: req.body.bmi ? parseFloat(req.body.bmi) : null,
        weightStatus: req.body.weightStatus || '',
        isCoughProductive: req.body.isCoughProductive || '',
        coughDurationDays: req.body.coughDurationDays ? parseInt(req.body.coughDurationDays) : null,
        hadPriorTB: req.body.hadPriorTB || '',
        tobaccoUse: req.body.tobaccoUse || '',
        cigarettesPerDay: req.body.cigarettesPerDay ? parseInt(req.body.cigarettesPerDay) : null,
        smokingSinceMonths: req.body.smokingSinceMonths ? parseInt(req.body.smokingSinceMonths) : null,
        stoppedSmokingMonths: req.body.stoppedSmokingMonths ? parseInt(req.body.stoppedSmokingMonths) : null,
        comorbidities: Array.isArray(req.body.comorbidities) ? req.body.comorbidities : (req.body.comorbidities ? [req.body.comorbidities] : []),
        comorbiditiesOther: req.body.comorbiditiesOther || ''
      };

      const savedProfile = await TbcareProfile.create(profileData);
      savedUser.tbcareProfile = savedProfile._id;
      await savedUser.save();
    } catch (e) {
      console.error('Failed to create tbcare profile:', e);
    }

    res.redirect("/sub_1/doctor");

  } catch (err) {
    console.log(err);
    req.flash('error', 'Failed to create patient. Email may already exist.');
    res.redirect('/sub_1/add-patient');
  }
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
      provinces: regions.provinces,
      surabayaDistricts: regions.surabayaDistricts,
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
    // update basic user fields
    await User.updateOne({ _id: patientId, doctor: req.session.user._id }, {
      $set: {
        'fullName.first': fname,
        'fullName.last': lname,
        email: email,
        userName: uname
      }
    });

    // compute age from dateOfBirth if provided
    const computedAge = calculateAge(req.body.dateOfBirth);

    // normalize/parsing values
    const parsedHeight = height ? parseFloat(height) : null;
    const parsedWeight = weight ? parseFloat(weight) : null;
    const parsedBmi = bmi ? parseFloat(bmi) : (parsedHeight && parsedWeight ? +(parsedWeight / ((parsedHeight/100)*(parsedHeight/100))).toFixed(2) : null);
    const parsedCoughDuration = coughDurationDays ? parseInt(coughDurationDays) : null;
    const parsedWeightLoss = weightLossAmountKg ? parseFloat(weightLossAmountKg) : null;
    const parsedCigarettes = cigarettesPerDay ? parseInt(cigarettesPerDay) : null;
    const parsedSmokingSince = smokingSinceMonths ? parseInt(smokingSinceMonths) : null;
    const parsedStoppedSmoking = stoppedSmokingMonths ? parseInt(stoppedSmokingMonths) : null;

    // boolean checkboxes (checkbox may be 'on' or 'true' or undefined)
    const bool = v => (typeof v !== 'undefined' && v !== null && (v === 'on' || v === 'true' || v === true));

    const parsedHasHemoptysis = bool(hasHemoptysis);
    const parsedHasChestPain = bool(hasChestPain);
    const parsedHasShortBreath = bool(hasShortBreath);
    const parsedHasFever = bool(hasFever);
    const parsedHasNightSweats = bool(hasNightSweats);
    const parsedHasWeightLoss = bool(hasWeightLoss);

    // comorbidities: accept string (comma separated) or array
    let parsedComorbidities = [];
    if (Array.isArray(comorbidities)) parsedComorbidities = comorbidities.map(c => (c || '').toString().trim()).filter(Boolean);
    else if (typeof comorbidities === 'string' && comorbidities.trim() !== '') {
      parsedComorbidities = comorbidities.split(',').map(c => c.trim()).filter(Boolean);
    }

    // upsert profile: jika belum ada maka buat
    await TbcareProfile.updateOne(
      { user: patientId },
      {
        $set: {
          province: req.body.province || null,
          city: req.body.city || null,
          district: req.body.district || null,
          dateOfBirth: req.body.dateOfBirth || null,
          sex: sex || null,
          age: computedAge || null,
          height: parsedHeight,
          weight: parsedWeight,
          bmi: parsedBmi,
          weightStatus: weightStatus || null,
          isCoughProductive: isCoughProductive || null,
          coughDurationDays: parsedCoughDuration,
          hasHemoptysis: parsedHasHemoptysis,
          hasChestPain: parsedHasChestPain,
          hasShortBreath: parsedHasShortBreath,
          hasFever: parsedHasFever,
          hasNightSweats: parsedHasNightSweats,
          hasWeightLoss: parsedHasWeightLoss,
          weightLossAmountKg: parsedWeightLoss,
          tobaccoUse: tobaccoUse || null,
          cigarettesPerDay: parsedCigarettes,
          smokingSinceMonths: parsedSmokingSince,
          stoppedSmokingMonths: parsedStoppedSmoking,
          hadPriorTB: hadPriorTB || null,
          comorbidities: parsedComorbidities,
          comorbiditiesOther: comorbiditiesOther || null
        }
      },
      { upsert: true }
    );

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
