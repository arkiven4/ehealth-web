const fs = require('fs');
const path = require('path');
const { PythonShell } = require('python-shell');
const { spawn } = require('child_process');

const User = require('../models/user');
const TbcarePrediction = require('../models/tbcare_prediction');
const TbcareProfile = require('../models/tbcare_profile');
const DeviceDataCoughTBPrimer = require("../models/device_data_cough_tbprimer");

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
            coughFiles: allFiles,
            audioFolders: [],
            csrfToken: req.csrfToken(),
            errorMessage: req.flash('error')[0]
        });
    } catch (error) {
        console.log("Error in getPredict:", error);
        next(error);
    }
};
async function runPythonScript(pythonPath, args = []) {
    return new Promise((resolve, reject) => {
        const py = spawn('python3', [pythonPath, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
        let out = '';
        let err = '';

        py.stdout.on('data', (d) => {
            out += d.toString();
        });

        py.stderr.on('data', (d) => {
            err += d.toString();
        });

        py.on('close', (code) => {
            if (code !== 0 && !out) {
                return reject(new Error(`Python script exited with code ${code}. stderr: ${err}`));
            }

            try {
                const lines = out.trim().split('\n');
                const jsonLine = lines.reverse().find(line =>
                    line.trim().startsWith('{') && line.trim().endsWith('}')
                );

                if (!jsonLine) {
                    throw new Error(`Tidak ditemukan JSON pada output Python.\nOutput:\n${out}\nStderr:\n${err}`);
                }

                const parsed = JSON.parse(jsonLine);
                resolve({ parsed, stdout: out, stderr: err, code });
            } catch (e) {
                reject(new Error(`Gagal parsing output Python.\nstdout:\n${out}\nstderr:\n${err}\n\n${e}`));
            }
        });

        py.on('error', (e) => reject(e));
    });
}

/**
 * @description Memproses form prediksi, menjalankan Python, menyimpan riwayat, dan menampilkan hasilnya.
 */
exports.postPredict = async (req, res, next) => {
    try {
        const { patientId, coughFilePath, sputumStatus } = req.body;

        if (!patientId || !coughFilePath || !sputumStatus) {
            req.flash('error', 'Harap lengkapi semua kolom.');
            return res.redirect('/tbcare/predict');
        }

        const fileSystemPath = path.join(__dirname, '..', 'public', coughFilePath);
        if (!fs.existsSync(fileSystemPath)) {
            req.flash('error', 'File audio tidak ditemukan.');
            return res.redirect('/tbcare/predict');
        }

        const pythonScriptPath = path.join(__dirname, '..', 'python-script', 'tbcareScript.py');
        const result = await runPythonScript(pythonScriptPath, [fileSystemPath]);
        const parsed = result.parsed;

        if (!parsed || parsed.status === 'error') {
            req.flash('error', parsed?.message || 'Skrip Python gagal.');
            return res.redirect('/tbcare/predict');
        }

        const patient = await User.findById(patientId).populate('tbcareProfile');
        if (!patient) {
            req.flash('error', 'Pasien tidak ditemukan.');
            return res.redirect('/tbcare/predict');
        }

        const tbSegments = parsed.tb_segments ?? 0;
        const nonTbSegments = parsed.non_tb_segments ?? 0;
        const totalSegments = parsed.total_segments ?? 0;

        await TbcarePrediction.create({
            patient: patientId,
            predictedBy: req.session.user._id,
            audioFile: coughFilePath,
            sputumCondition: sputumStatus,
            result: parsed.final_decision,
            tbSegmentCount: tbSegments,
            nonTbSegmentCount: nonTbSegments,
            totalCoughSegments: totalSegments,
        });

        return res.render('doctor/tbcare/predict-result', {
            pageTitle: 'Hasil Prediksi',
            pageHeader: 'Hasil Prediksi',
            userdata: req.session.user,
            patient,
            prediction: parsed.final_decision,
            detail: {
                tb_segments: tbSegments,
                non_tb_segments: nonTbSegments,
                total_segments: totalSegments,
            },
            sputumCondition: sputumStatus,
            audioFile: coughFilePath,
            waveform: JSON.stringify(parsed.waveform || []),
            mfcc: JSON.stringify(parsed.mfcc || []),
            csrfToken: req.csrfToken(),
        });

    } catch (err) {
        console.error("=== ERROR DETAIL postPredict ===");
        console.error("Message:", err.message);
        console.error("Stack:", err.stack);

        if (!res.headersSent) {
            req.flash('error', `Terjadi kesalahan sistem: ${err.message}`);
            return res.redirect('/tbcare/predict');
        }
    }
};

exports.getPatientHistoryList = async (req, res, next) => {
    try {
        const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate('tbcareProfile');
        res.render('doctor/tbcare/patient-history-list', {
            pageTitle: 'Patient History',
            pageHeader: 'Patient Prediction History',
            userdata: req.session.user,
            patients: patients,
            csrfToken: req.csrfToken()
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