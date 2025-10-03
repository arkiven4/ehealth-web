exports.getLandingPage = (req, res, next) => {
  res.render('tbcare/landing', {
    pageTitle: 'Welcome to TBCare',
    isAuthenticated: req.session.isLoggedIn 
  });
};

exports.getDownloadPage = (req, res, next) => {
  res.render('tbcare/download', {
    pageTitle: 'Download TBCare App',
    isAuthenticated: req.session.isLoggedIn
  });
};

exports.getAboutPage = (req, res, next) => {
  res.render('tbcare/about', {
    pageTitle: 'About TBCare',
    isAuthenticated: req.session.isLoggedIn
  });
};

exports.getTbcareLoginPage = (req, res, next) => {
  res.render('tbcare/login', {
    pageTitle: 'TBCare Login',
    csrfToken: req.csrfToken() 
  });
};

const { PythonShell } = require('python-shell');
const path = require('path');
const TbcarePrediction = require('../models/tbcare_prediction');
const TbcareProfile = require('../models/tbcare_profile');

// FUNGSI INI DIPERBAIKI DAN DIAKTIFKAN KEMBALI
exports.getPredict = (req, res, next) => {
    res.render('doctor/tbcare/predict', {
        pageTitle: 'Predict Cough Audio',
        path: '/tbcare/predict',
        hasResult: false,
        errorMessage: null,
        // Tambahkan variabel ini agar tidak error saat halaman pertama kali dimuat
        predictionResult: null,
        predictionDetail: null,
        waveform: null,
        mfcc: null,
    });
};

exports.postPredict = (req, res, next) => {
    // 1. Validasi apakah ada file yang diunggah
    if (!req.file) {
        return res.status(400).render('doctor/tbcare/predict', {
            pageTitle: 'Predict Cough Audio',
            path: '/tbcare/predict',
            hasResult: false,
            errorMessage: 'Tidak ada file yang diunggah. Silakan pilih file audio.',
            predictionResult: null,
            predictionDetail: null,
            waveform: null,
            mfcc: null,
        });
    }

    const audioFilePath = req.file.path;

    const options = {
        mode: 'text',
        pythonPath: 'python3', 
        scriptPath: path.join(__dirname, '..', 'python-script'),
        args: [audioFilePath]
    };

    PythonShell.run('tbcareScript.py', options).then(results => {
        try {
            const data = JSON.parse(results[0]);

            if (data.status === 'error') {
                console.error('Python Script Error:', data.message);
                return res.status(500).render('doctor/tbcare/predict', {
                    pageTitle: 'Prediction Error',
                    path: '/tbcare/predict',
                    hasResult: false,
                    errorMessage: data.message,
                    predictionResult: null,
                    predictionDetail: null,
                    waveform: null,
                    mfcc: null,
                });
            }

            res.render('doctor/tbcare/predict', {
                pageTitle: 'Prediction Result',
                path: '/tbcare/predict',
                hasResult: true,
                errorMessage: null,
                predictionResult: data.prediction,
                predictionDetail: data.detail,
                waveform: JSON.stringify(data.waveform),
                mfcc: JSON.stringify(data.mfcc),
            });

        } catch (parseError) {
            console.error('JSON Parse Error:', parseError, 'Python Output:', results);
            res.status(500).render('doctor/tbcare/predict', {
                pageTitle: 'Prediction Error',
                path: '/tbcare/predict',
                hasResult: false,
                errorMessage: 'Gagal memproses hasil dari skrip analisis. Output tidak valid.',
                predictionResult: null,
                predictionDetail: null,
                waveform: null,
                mfcc: null,
            });
        }
    }).catch(err => {
        console.error('PythonShell Execution Error:', err);
        res.status(500).render('doctor/tbcare/predict', {
            pageTitle: 'Execution Error',
            path: '/tbcare/predict',
            hasResult: false,
            errorMessage: 'Terjadi kesalahan sistem saat mencoba menjalankan skrip analisis audio.',
            predictionResult: null,
            predictionDetail: null,
            waveform: null,
            mfcc: null,
        });
    });
};

exports.savePrediction = async (req, res, next) => {
    try {
        const {
            patientId,
            audioFile,
            sputumCondition,
            result,
            tbSegmentCount,
            nonTbSegmentCount,
            totalCoughSegments
        } = req.body;

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
        console.error('Save Prediction Error:', err);
        req.flash('error', 'Gagal menyimpan hasil prediksi ke database.');
        res.redirect('/tbcare/predict');
    }
};