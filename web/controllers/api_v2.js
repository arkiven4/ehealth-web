const path = require('path');
const fs = require('fs');
const { PythonShell } = require('python-shell');

// Panggil semua model yang kita butuhkan
const TbcareProfile = require('../models/tbcare_profile');
const TbcarePrediction = require('../models/tbcare_prediction');
const User = require('../models/user');

/**
 * @description Membuat entri prediksi baru, menjalankan script Python, dan menyimpan hasilnya.
 */
exports.postStartPrediction = async (req, res, next) => {
    try {
        const { participantId, filename, sputumStatus, sputumLevel } = req.body;
        
        if (!participantId || !filename || !sputumStatus) {
            return res.status(400).json({ 
                message: 'Operasi Gagal: participantId, filename, dan sputumStatus wajib diisi.'
            });
        }

        const patientProfile = await TbcareProfile.findOne({ participantId });
        if (!patientProfile) {
            return res.status(404).json({ 
                message: `Pasien dengan participantId ${participantId} tidak ditemukan.` 
            });
        }

        const patientUser = await User.findById(patientProfile.user);
        const doctorUser = await User.findOne({ role: 'doctor' }); // Ambil dokter mana saja untuk referensi
        
        const filePath = path.join(__dirname, '..', 'public', 'uploads', 'batuk_tbprimer', filename);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ 
                message: `File audio tidak ditemukan di server: ${filePath}`
            });
        }

        let sputumConditionLabel = sputumStatus;
        if (sputumStatus === 'Sputum +' && sputumLevel) {
            sputumConditionLabel += ` (${sputumLevel})`;
        }

        const options = {
            mode: 'text',
            pythonPath: 'python3',
            scriptPath: path.join(__dirname, '..', 'python-script'),
            args: [filePath]
        };

        PythonShell.run('tbcareScript.py', options, async (err, results) => {
            if (err) {
                console.error('PythonShell execution error:', err);
                return res.status(500).json({ message: 'Gagal menjalankan skrip analisis Python.' });
            }

            try {
                const jsonString = results[0].replace(/'/g, '"');
                const dataFromPython = JSON.parse(jsonString);

                if (dataFromPython.status === 'error') {
                    return res.status(500).json({ message: `Prediksi gagal: ${dataFromPython.message}` });
                }

                const details = dataFromPython.detail;
                const confidenceScore = details.total_segments > 0 
                    ? (details.tb_segments / details.total_segments) 
                    : 0;
                
                const completeDetailObject = {
                    ...details,
                    waveform: dataFromPython.waveform,
                    mfcc: dataFromPython.mfcc
                };

                const newPrediction = new TbcarePrediction({
                    patient: patientUser._id,
                    predictedBy: doctorUser ? doctorUser._id : patientUser._id,
                    audioFile: `/uploads/batuk_tbprimer/${filename}`,
                    sputumCondition: sputumConditionLabel,
                    result: dataFromPython.prediction,
                    confidence: confidenceScore,
                    tbSegmentCount: details.tb_segments,
                    nonTbSegmentCount: details.non_tb_segments,
                    totalCoughSegments: details.total_segments,
                    detail: completeDetailObject
                });

                await newPrediction.save();

                res.status(201).json({
                    message: 'Prediksi berhasil disimpan dengan data visualisasi!',
                    predictionId: newPrediction._id,
                    result: newPrediction.result,
                    confidence: newPrediction.confidence
                });

            } catch (parseError) {
                console.error('Failed to parse JSON from Python script:', parseError);
                console.error('Original Python output:', results);
                return res.status(500).json({ message: 'Gagal memproses hasil dari skrip analisis.' });
            }
        });

    } catch (error) {
        console.error('ERROR di /api/v2/start-prediction:', error);
        res.status(500).json({
            message: 'Terjadi kesalahan internal di server.',
            error_code: 'V2_START_PREDICTION_FAILED',
            details: error.message
        });
    }
};

/**
 * @description Membuat data User dan TbcareProfile baru untuk keperluan testing.
 */
exports.postCreateTestPatient = async (req, res, next) => {
    try {
        const timestamp = Date.now();
        
        const newTestUser = new User({
            email: `testuser-${timestamp}@tbcare.dev`,
            userName: `testuser-${timestamp}`,
            password: 'password_test_123',
            accountType: 'tbcare',
            role: 'patient'
        });
        const savedUser = await newTestUser.save();

        const newTestProfile = new TbcareProfile({
            user: savedUser._id,
            participantId: `TEST-PATIENT-${timestamp}`,
            sex: 'Female',
            age: 35
        });
        const savedProfile = await newTestProfile.save();

        savedUser.tbcareProfile = savedProfile._id;
        await savedUser.save();
        
        res.status(201).json({
            message: 'Pasien Uji Coba LENGKAP (User + Profile) BERHASIL dibuat!',
            note: 'Gunakan participantId di bawah ini untuk testing /start-prediction',
            testPatientData: {
                userId: savedUser._id,
                userName: savedUser.userName,
                profileId: savedProfile._id,
                participantId: savedProfile.participantId
            }
        });

    } catch (error) {
        console.error('ERROR DI /api/v2/create-test-patient:', error);
        res.status(500).json({
            message: 'Gagal membuat pasien uji coba.',
            error_code: 'V2_CREATE_PATIENT_FAILED',
            details: error.message
        });
    }
};

/**
 * @description Mengambil semua riwayat prediksi milik satu pasien.
 */
exports.getPatientHistory = async (req, res, next) => {
    try {
        const { participantId } = req.params;

        const patientProfile = await TbcareProfile.findOne({ participantId });
        if (!patientProfile) {
            return res.status(404).json({ message: 'Pasien tidak ditemukan.' });
        }

        const predictions = await TbcarePrediction.find({ patient: patientProfile.user })
            .sort({ createdAt: -1 });

        res.status(200).json({
            message: `Berhasil mengambil ${predictions.length} data riwayat untuk pasien ${participantId}.`,
            history: predictions
        });

    } catch (error) {
        console.error('ERROR di /api/v2/patient-history:', error);
        res.status(500).json({ message: 'Gagal mengambil data history.', details: error.message });
    }
};

/**
 * @description Mengubah data prediksi, seperti mapping pasien atau sputum condition.
 */
exports.updatePrediction = async (req, res, next) => {
    try {
        const { predictionId } = req.params;
        const { newParticipantId, sputumStatus, sputumLevel } = req.body;

        const prediction = await TbcarePrediction.findById(predictionId);
        if (!prediction) {
            return res.status(404).json({ message: 'Data prediksi tidak ditemukan.' });
        }

        if (newParticipantId) {
            const newPatientProfile = await TbcareProfile.findOne({ participantId: newParticipantId });
            if (!newPatientProfile) {
                return res.status(404).json({ message: `Pasien baru dengan ID ${newParticipantId} tidak ditemukan.` });
            }
            prediction.patient = newPatientProfile.user;
        }
        
        if (sputumStatus) {
            let sputumConditionLabel = sputumStatus;
            if (sputumStatus === 'Sputum +' && sputumLevel) {
                sputumConditionLabel += ` (${sputumLevel})`;
            }
            prediction.sputumCondition = sputumConditionLabel;
        }

        const updatedPrediction = await prediction.save();

        res.status(200).json({
            message: 'Data prediksi berhasil diperbarui!',
            data: updatedPrediction
        });

    } catch (error) {
        console.error('ERROR di /api/v2/prediction/:predictionId (UPDATE):', error);
        res.status(500).json({ message: 'Gagal memperbarui data prediksi.', details: error.message });
    }
};

/**
 * @description Menghapus satu data prediksi dari database.
 */
exports.deletePrediction = async (req, res, next) => {
    try {
        const { predictionId } = req.params;

        const result = await TbcarePrediction.findByIdAndDelete(predictionId);

        if (!result) {
            return res.status(404).json({ message: 'Gagal menghapus: Data prediksi tidak ditemukan.' });
        }

        res.status(200).json({ 
            message: 'Data prediksi berhasil dihapus.',
            deletedData: result
        });

    } catch (error) {
        console.error('ERROR di /api/v2/prediction/:predictionId (DELETE):', error);
        res.status(500).json({ message: 'Gagal menghapus data prediksi.', details: error.message });
    }
};