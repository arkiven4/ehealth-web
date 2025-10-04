const fs = require('fs');
const path = require('path');
const { PythonShell } = require('python-shell');

const User = require('../models/user');
const TbcarePrediction = require('../models/tbcare_prediction');

// Fungsi-fungsi landing page tidak berubah
exports.getLandingPage = (req, res, next) => {
  res.render('tbcare/landing', { pageTitle: 'Welcome to TBCare', isAuthenticated: req.session.isLoggedIn });
};
exports.getDownloadPage = (req, res, next) => {
  res.render('tbcare/download', { pageTitle: 'Download TBCare App', isAuthenticated: req.session.isLoggedIn });
};
exports.getAboutPage = (req, res, next) => {
  res.render('tbcare/about', { pageTitle: 'About TBCare', isAuthenticated: req.session.isLoggedIn });
};
exports.getTbcareLoginPage = (req, res, next) => {
  res.render('tbcare/login', { pageTitle: 'TBCare Login', csrfToken: req.csrfToken() });
};

// --- LOGIKA PREDIKSI ---

/**
 * GET /predict
 * Menampilkan halaman formulir prediksi dengan daftar pasien dan file audio.
 */
exports.getPredict = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate('tbcareProfile');
    const uploadsPath = path.join(__dirname, '..', 'public', 'uploads', 'batuk_tbprimer');
    let allFiles = [];

    if (fs.existsSync(uploadsPath)) {
      allFiles = fs.readdirSync(uploadsPath)
        .filter(file => file.endsWith('.wav'))
        .map(file => {
          const stats = fs.statSync(path.join(uploadsPath, file));
          return {
            name: file,
            path: `/uploads/batuk_tbprimer/${file}`,
            modifiedTime: stats.mtime.getTime(),
            displayDate: stats.mtime.toISOString().split('T')[0],
            folder: ''
          };
        });
    }
    allFiles.sort((a, b) => b.modifiedTime - a.modifiedTime);

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction",
      pageHeader: "Cough Recording Prediction",
      userdata: req.session.user,
      patients: patients,
      coughFiles: allFiles,
      audioFolders: [],
      csrfToken: req.csrfToken(),
      errorMessage: req.flash('error')[0],
      // Menambahkan variabel dummy agar halaman tidak error saat pertama kali render
      hasResult: false,
      predictionResult: null,
      predictionDetail: null,
      waveform: null,
      mfcc: null,
    });
  } catch (error) {
    console.log("Error in getPredict:", error);
    next(error);
  }
};

/**
 * POST /predict
 * Memproses data, menjalankan skrip Python, dan menampilkan hasil.
 */
exports.postPredict = async (req, res, next) => {
  const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

  if (!patientId || !coughFilePath || !sputumStatus) {
    req.flash('error', 'Harap lengkapi semua kolom yang diperlukan.');
    return res.redirect('/tbcare/predict');
  }

  // Membuat path absolut dari file sistem berdasarkan path URL yang diterima
  const fileSystemPath = path.join(__dirname, '..', 'public', coughFilePath);

  // Validasi: Cek apakah file benar-benar ada sebelum menjalankan skrip
  if (!fs.existsSync(fileSystemPath)) {
    console.error(`File not found at path: ${fileSystemPath}`);
    req.flash('error', `File audio tidak ditemukan di server: ${coughFilePath}`);
    return res.redirect('/tbcare/predict');
  }

  const options = {
    mode: 'text',
    pythonPath: 'python3',
    scriptPath: path.join(__dirname, '..', 'python-script'),
    args: [fileSystemPath]
  };

  PythonShell.run('tbcareScript.py', options, async (err, results) => {
    if (err) {
      console.error('PythonShell execution error:', err);
      req.flash('error', 'Terjadi kesalahan sistem saat menjalankan skrip analisis.');
      return res.redirect('/tbcare/predict');
    }

    if (!results || results.length === 0 || !results[0]) {
      console.error('Invalid/empty output from Python script:', results);
      req.flash('error', 'Skrip analisis tidak memberikan output yang valid.');
      return res.redirect('/tbcare/predict');
    }

    try {
      const jsonString = results[0].replace(/'/g, '"');
      const data = JSON.parse(jsonString);

      if (data.status === 'error') {
        console.error('Error message from Python script:', data.message);
        req.flash('error', `Prediksi gagal: ${data.message}`);
        return res.redirect('/tbcare/predict');
      }

      const patient = await User.findById(patientId).populate('tbcareProfile');
      if (!patient) {
        req.flash('error', 'Pasien tidak ditemukan.');
        return res.redirect('/tbcare/predict');
      }

      let sputumConditionLabel = sputumStatus;
      if (sputumStatus === 'Sputum +' && sputumLevel) {
        sputumConditionLabel += ` (${sputumLevel})`;
      }

      res.render('doctor/tbcare/predict-result', {
        pageTitle: 'Hasil Prediksi',
        pageHeader: 'Hasil Prediksi',
        userdata: req.session.user,
        csrfToken: req.csrfToken(),
        patient: patient,
        prediction: data.prediction,
        detail: data.detail,
        sputumCondition: sputumConditionLabel,
        audioFile: coughFilePath,
        waveform: JSON.stringify(data.waveform),
        mfcc: JSON.stringify(data.mfcc)
      });

    } catch (parseError) {
      console.error('Failed to parse JSON from Python script:', parseError);
      console.error('Original Python output:', results);
      req.flash('error', 'Gagal memproses hasil dari skrip analisis.');
      return res.redirect('/tbcare/predict');
    }
  });
};


/**
 * POST /save-prediction
 * Menerima data dari halaman hasil dan menyimpannya ke database.
 */
exports.savePrediction = async (req, res, next) => {
  try {
    const { patientId, audioFile, sputumCondition, result, tbSegmentCount, nonTbSegmentCount, totalCoughSegments } = req.body;
    await TbcarePrediction.create({
      user: patientId,
      predictedBy: req.session.user._id,
      audioFile: audioFile,
      sputumCondition: sputumCondition,
      result: result,
      tbSegmentCount: parseInt(tbSegmentCount),
      nonTbSegmentCount: parseInt(nonTbSegmentCount),
      totalCoughSegments: parseInt(totalCoughSegments)
    });
    req.flash('success', `Hasil prediksi untuk file ${audioFile.split('/').pop()} berhasil disimpan.`);
    res.redirect("/sub_1/doctor");
  } catch (err) {
    console.error("Save Prediction Error:", err);
    req.flash('error', 'Gagal menyimpan hasil prediksi ke database.');
    res.redirect('/tbcare/predict');
  }
};