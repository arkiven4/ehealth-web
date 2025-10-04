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

// --- LOGIKA PREDIKSI BARU ---

/**
 * GET /predict
 * Menampilkan halaman formulir prediksi dengan daftar pasien dan file audio.
 */
exports.getPredict = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate('tbcareProfile');
    const uploadsPath = path.join(__dirname, '..', 'public', 'uploads', 'tbcare');
    let allFiles = [];
    if (fs.existsSync(uploadsPath)) {
      const mainFolders = fs.readdirSync(uploadsPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
      mainFolders.forEach(folder => {
        const folderPath = path.join(uploadsPath, folder);
        const files = fs.readdirSync(folderPath).filter(file => file.endsWith('.wav')).map(file => {
          const stats = fs.statSync(path.join(folderPath, file));
          return {
            name: file, folder: folder, path: `/uploads/tbcare/${folder}/${file}`,
            modifiedTime: stats.mtime.getTime(), displayDate: stats.mtime.toISOString().split('T')[0]
          };
        });
        allFiles.push(...files);
      });
    }
    allFiles.sort((a, b) => b.modifiedTime - a.modifiedTime);

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction", pageHeader: "Cough Recording Prediction",
      userdata: req.session.user, patients: patients, coughFiles: allFiles,
      audioFolders: [...new Set(allFiles.map(f => f.folder))], // Ambil folder unik
      csrfToken: req.csrfToken(), errorMessage: req.flash('error')[0]
    });
  } catch (error) {
    console.log(error);
    next(error);
  }
};

/**
 * POST /predict
 * Memproses data dari formulir, menjalankan skrip Python, dan menampilkan halaman hasil.
 */
exports.postPredict = async (req, res, next) => {
  const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

  if (!patientId || !coughFilePath || !sputumStatus) {
    req.flash('error', 'Harap lengkapi semua kolom yang diperlukan.');
    return res.redirect('/tbcare/predict');
  }

  const relativePath = coughFilePath.startsWith('/') ? coughFilePath.substring(1) : coughFilePath;
  const coughFileAbsolutePath = path.join(__dirname, '..', 'public', relativePath);

  try {
    PythonShell.run("./python-script/tbcareScript.py", { args: ["/usr/src/app/public/" + relativePath] }, async function (err, results) {
      if (err) throw err;
      const data = JSON.parse(results[2]);
      if (data.status === 'error') {
        req.flash('error', `Prediksi gagal dari skrip: ${data.message}`);
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
        pageTitle: 'Hasil Prediksi', pageHeader: 'Hasil Prediksi',
        userdata: req.session.user, csrfToken: req.csrfToken(),
        patient: patient, prediction: data.prediction, detail: data.detail,
        sputumCondition: sputumConditionLabel, audioFile: coughFilePath,
        waveform: JSON.stringify(data.waveform), mfcc: JSON.stringify(data.mfcc)
      });
    });
    // // Perbaikan: Cek apakah skrip Python memberikan output sebelum parsing
    // if (!results || results.length === 0 || !results[0]) {
    //   // throw new Error('Skrip Python tidak memberikan output yang valid.');
    // }
  } catch (err) {
    console.error("Prediction Error:", err);
    // Kirim pesan error yang lebih spesifik jika ada
    const errorMessage = err.message || 'Terjadi kesalahan sistem saat menjalankan atau memproses hasil prediksi.';
    req.flash('error', errorMessage);
    res.redirect('/tbcare/predict');
  }
};


/**
 * POST /save-prediction
 * Menerima data dari halaman hasil dan menyimpannya ke database.
 */
exports.savePrediction = async (req, res, next) => {
  try {
    const { patientId, audioFile, sputumCondition, result, tbSegmentCount, nonTbSegmentCount, totalCoughSegments } = req.body;
    await TbcarePrediction.create({
      user: patientId, predictedBy: req.session.user._id,
      audioFile: audioFile, sputumCondition: sputumCondition, result: result,
      tbSegmentCount: parseInt(tbSegmentCount), nonTbSegmentCount: parseInt(nonTbSegmentCount),
      totalCoughSegments: parseInt(totalCoughSegments)
    });
    req.flash('success', `Hasil prediksi untuk file ${audioFile.split('/').pop()} berhasil disimpan.`);
    res.redirect("/sub_1/doctor");
  } catch (err) {
    req.flash('error', 'Gagal menyimpan hasil prediksi ke database.');
    res.redirect('/tbcare/predict');
  }
};