const fs = require('fs');
const path = require('path');
const { PythonShell } = require('python-shell');

const User = require('../models/user');
const TbcarePrediction = require('../models/tbcare_prediction');
const TbcareProfile = require('../models/tbcare_profile');

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

/**
 * @description Menampilkan halaman untuk memulai prediksi baru (menampilkan semua file audio).
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
            coughFiles: allFiles, // <-- Diubah di sini: Langsung gunakan 'allFiles' tanpa difilter
            audioFolders: [],
            csrfToken: req.csrfToken(),
            errorMessage: req.flash('error')[0]
        });
    } catch (error) {
        console.log("Error in getPredict:", error);
        next(error);
    }
};

/**
 * @description (VERSI FINAL) Memproses form prediksi, menjalankan Python, menyimpan riwayat, dan menampilkan hasilnya.
 */
exports.postPredict = async (req, res, next) => {
    const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

    if (!patientId || !coughFilePath || !sputumStatus) {
        req.flash('error', 'Harap lengkapi semua kolom yang diperlukan.');
        return res.redirect('/tbcare/predict');
    }
    const fileSystemPath = path.join(__dirname, '..', 'public', coughFilePath);
    if (!fs.existsSync(fileSystemPath)) {
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
        if (err || !results || results.length === 0) {
            console.error('PythonShell execution error:', err || 'No results from script');
            req.flash('error', 'Terjadi kesalahan sistem saat menjalankan skrip analisis.');
            return res.redirect('/tbcare/predict');
        }

        try {
            const jsonString = results[0].replace(/'/g, '"');
            const dataFromPython = JSON.parse(jsonString);

            if (dataFromPython.status === 'error') {
                req.flash('error', `Prediksi gagal: ${dataFromPython.message}`);
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
            
            const details = dataFromPython.detail;
            const confidenceScore = details.total_segments > 0 
                ? (details.tb_segments / details.total_segments) 
                : 0;

            const newPrediction = new TbcarePrediction({
                patient: patient._id,
                predictedBy: req.session.user._id,
                audioFile: coughFilePath,
                sputumCondition: sputumConditionLabel,
                result: dataFromPython.prediction,
                confidence: confidenceScore,
                tbSegmentCount: details.tb_segments,
                nonTbSegmentCount: details.non_tb_segments,
                totalCoughSegments: details.total_segments,
                detail: { ...details, waveform: dataFromPython.waveform, mfcc: dataFromPython.mfcc }
            });
            await newPrediction.save();

            res.render('doctor/tbcare/predict-result', {
                pageTitle: 'Hasil Prediksi',
                pageHeader: 'Hasil Prediksi',
                userdata: req.session.user,
                csrfToken: req.csrfToken(),
                patient: patient,
                prediction: dataFromPython.prediction,
                detail: dataFromPython.detail,
                sputumCondition: sputumConditionLabel,
                audioFile: coughFilePath,
                waveform: JSON.stringify(dataFromPython.waveform),
                mfcc: JSON.stringify(dataFromPython.mfcc)
            });

        } catch (parseError) {
            console.error('Failed to parse JSON from Python script:', parseError);
            req.flash('error', 'Gagal memproses hasil dari skrip analisis.');
            return res.redirect('/tbcare/predict');
        }
    });
};

/**
 * @description Menampilkan daftar semua pasien dalam bentuk kartu untuk halaman riwayat.
 */
exports.getPatientHistoryList = async (req, res, next) => {
    try {
        const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate('tbcareProfile');
        res.render('doctor/tbcare/patient-history-list', {
            pageTitle: 'Patient History',
            pageHeader: 'Patient Prediction History',
            userdata: req.session.user,
            patients: patients
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @description Menampilkan detail riwayat prediksi untuk satu pasien dalam bentuk tabel.
 */
exports.getPatientHistoryDetail = async (req, res, next) => {
    try {
        const patientId = req.params.patientId;
        const patient = await User.findById(patientId).populate('tbcareProfile');
        
        if (!patient) {
            return res.redirect('/tbcare/patient-history');
        }

        const predictions = await TbcarePrediction.find({ patient: patientId })
            .populate('predictedBy', 'userName')
            .sort({ createdAt: -1 });

        res.render('doctor/tbcare/patient-history-detail', {
            pageTitle: `History for ${patient.fullName.first}`,
            pageHeader: 'Prediction History',
            userdata: req.session.user,
            patient: patient,
            predictions: predictions,
            csrfToken: req.csrfToken()
        });
    } catch (error) {
        next(error);
    }
};