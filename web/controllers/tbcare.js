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

// exports.getPredict = (req, res, next) => {
//     res.render('doctor/tbcare/predict', {
//         pageTitle: 'Predict Cough Audio',
//         path: '/tbcare/predict',
//         hasResult: false,
//         errorMessage: null,
//     });
// };

exports.postPredict = (req, res, next) => {
    // 1. Validasi apakah ada file yang diunggah
    if (!req.file) {
        return res.status(400).render('doctor/tbcare/predict', {
            pageTitle: 'Predict Cough Audio',
            path: '/tbcare/predict',
            hasResult: false,
            errorMessage: 'Tidak ada file yang diunggah. Silakan pilih file audio.',
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
            });
        }
    }).catch(err => {
        console.error('PythonShell Execution Error:', err);
        res.status(500).render('doctor/tbcare/predict', {
            pageTitle: 'Execution Error',
            path: '/tbcare/predict',
            hasResult: false,
            errorMessage: 'Terjadi kesalahan sistem saat mencoba menjalankan skrip analisis audio.',
        });
    });
};