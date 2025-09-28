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

// Menampilkan halaman awal untuk unggah file prediksi
exports.getPredict = (req, res, next) => {
    res.render('doctor/tbcare/predict', {
        pageTitle: 'Predict Cough Audio',
        path: '/tbcare/predict',
        hasResult: false, // Belum ada hasil saat halaman pertama kali dimuat
        errorMessage: null,
    });
};

// Menangani permintaan POST setelah file diunggah
exports.postPredict = (req, res, next) => {
    if (!req.file) {
        // Jika tidak ada file, render kembali halaman dengan pesan error
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
        // Sesuaikan 'python3' jika perlu, atau gunakan path absolut ke environment python Anda
        pythonPath: 'python3', 
        scriptPath: path.join(__dirname, '..', 'python-script'),
        args: [audioFilePath]
    };

    PythonShell.run('tbcareScript.py', options).then(results => {
        try {
            // Hasil dari skrip python adalah string JSON, jadi kita parse
            const data = JSON.parse(results[0]);

            // Jika skrip Python mengembalikan status error
            if (data.status === 'error') {
                return res.status(500).render('doctor/tbcare/predict', {
                    pageTitle: 'Prediction Error',
                    path: '/tbcare/predict',
                    hasResult: false,
                    errorMessage: data.message,
                });
            }
            
            // Render halaman hasil dengan semua data yang dibutuhkan
            res.render('doctor/tbcare/predict', {
                pageTitle: 'Prediction Result',
                path: '/tbcare/predict',
                hasResult: true,
                errorMessage: null,
                predictionResult: data.prediction,
                predictionDetail: data.detail,
                // Kirim data visualisasi sebagai string JSON ke view
                waveform: JSON.stringify(data.waveform), 
                mfcc: JSON.stringify(data.mfcc), 
            });

        } catch (parseError) {
            // Error jika output dari python bukan JSON yang valid
            console.error('Error parsing python script output:', parseError);
            res.status(500).render('doctor/tbcare/predict', {
                 pageTitle: 'Prediction Error',
                 path: '/tbcare/predict',
                 hasResult: false,
                 errorMessage: "Gagal membaca hasil prediksi dari skrip.",
            });
        }
    }).catch(err => {
        // Error jika skrip python itu sendiri gagal dieksekusi
        console.error('PythonShell execution error:', err);
        res.status(500).render('doctor/tbcare/predict', {
             pageTitle: 'Prediction Error',
             path: '/tbcare/predict',
             hasResult: false,
             errorMessage: "Terjadi kesalahan saat menjalankan skrip analisis audio.",
        });
    });
};