const path = require('path');
const fs = require('fs');
const TbcareProfile = require('../models/tbcare_profile');
const TbcarePrediction = require('../models/tbcare_prediction');
const User = require('../models/user');
const { PythonShell } = require('python-shell');

exports.postStartPrediction = async (req, res, next) => {
    try {
        const { participantId, filename, sputumStatus, sputumLevel } = req.body;
        if (!participantId || !filename || !sputumStatus) {
            return res.status(400).json({ 
                message: 'Operasi Gagal: participantId, filename, dan sputumStatus wajib diisi.'
            });
        }

        // cari profil pasien
        const patientProfile = await TbcareProfile.findOne({ participantId });
        if (!patientProfile) {
            return res.status(404).json({ 
                message: `Pasien dengan participantId ${participantId} tidak ditemukan.` 
            });
        }

        const patientUser = await User.findById(patientProfile.user);
        const doctorUser = await User.findOne({ role: 'doctor' });
        const uploadsPath = path.join(__dirname, '..', 'public', 'uploads', 'batuk_tbprimer');
        const filePath = path.join(uploadsPath, filename);

        console.log(`[DEBUG] Mencari file audio di: ${filePath}`);

        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ 
                message: `File audio tidak ditemukan di server: ${filePath}`
            });
        }

        let sputumConditionLabel = sputumStatus;
        // Jika statusnya "Sputum +" dan levelnya diisi, gabungkan keduanya
        if (sputumStatus === 'Sputum +' && sputumLevel) {
            sputumConditionLabel += ` (${sputumLevel})`;
        }

        const options = {
            mode: 'text',
            pythonPath: 'python3', // Sesuaikan jika perlu
            scriptPath: path.join(__dirname, '..', 'python-script'), // Path ke folder script python
            args: [filePath] // Kirim path file audio sebagai argumen
        };

        PythonShell.run('tbcareScript.py', options, async (err, results) => {
            if (err) {
                // Jika script python-nya error, kita tangkap di sini
                console.error('PythonShell execution error:', err);
                return res.status(500).json({ message: 'Gagal menjalankan skrip analisis Python.'});
            }

            try {
                // 'results' akan berisi output dari script python
                // Kita asumsikan outputnya adalah JSON string
                const jsonString = results[0].replace(/'/g, '"');
                const data = JSON.parse(jsonString);

                if (data.status === 'error') {
                     return res.status(500).json({ message: `Prediksi gagal: ${data.message}`});
                }
                
                // Ganti blok const newPrediction = new TbcarePrediction({...}); dengan ini:

                const newPrediction = new TbcarePrediction({
                    patient: patientUser._id,
                    predictedBy: doctorUser ? doctorUser._id : patientUser._id,
                    audioFile: `/uploads/batuk_tbprimer/${filename}`,
                    sputumCondition: sputumConditionLabel,
                    result: data.prediction,
                    confidence: parseFloat(data.detail.confidence),
                    
                    // --- PERBAIKAN NAMA KEY DENGAN 's' DI AKHIR ---
                    tbSegmentCount: data.detail.tb_segments,
                    nonTbSegmentCount: data.detail.non_tb_segments,
                    totalCoughSegments: data.detail.total_segments,
                    detail: data.detail 
                });
                await newPrediction.save();

                res.status(201).json({
                    message: 'Prediksi dari model AI berhasil dilakukan!',
                    data: {
                        patientName: patientUser ? patientUser.userName : 'N/A',
                        participantId: patientProfile.participantId,
                        predictionId: newPrediction._id,
                        result: newPrediction.result,
                        confidence: newPrediction.confidence
                    }
                });
                
            } catch (parseError) {
                console.error('Failed to parse JSON from Python script:', parseError);
                console.error('Original Python output:', results);
                return res.status(500).json({ message: 'Gagal memproses hasil dari skrip analisis.'});
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

// === FUNGSI BARU YANG LEBIH LENGKAP UNTUK MEMBUAT DATA UJI COBA ===
exports.postCreateTestPatient = async (req, res, next) => {
    try {
        const timestamp = Date.now();
        
        // 1. Buat User baru dengan data unik
        const newTestUser = new User({
            email: `testuser-${timestamp}@tbcare.dev`,
            userName: `testuser-${timestamp}`,
            password: 'password_test_123', // Password ini tidak akan di-hash, tidak apa-apa untuk testing
            accountType: 'tbcare',
            role: 'patient'
        });
        const savedUser = await newTestUser.save();

        // 2. Buat TbcareProfile baru dan hubungkan dengan user di atas
        const newTestProfile = new TbcareProfile({
            user: savedUser._id, // Hubungkan ke user yang baru dibuat
            participantId: `TEST-PATIENT-${timestamp}`,
            sex: 'Female',
            age: 35
        });
        const savedProfile = await newTestProfile.save();

        // 3. (Opsional tapi praktik yang baik) Simpan referensi profil kembali ke user
        savedUser.tbcareProfile = savedProfile._id;
        await savedUser.save();
        
        // 4. Kirim kembali data pasien yang baru dibuat
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

exports.getPatientHistory = async (req, res, next) => {
    try {
        const { participantId } = req.params;

        // 1. Cari profil pasien
        const patientProfile = await TbcareProfile.findOne({ participantId });
        if (!patientProfile) {
            return res.status(404).json({ message: 'Pasien tidak ditemukan.' });
        }

        // 2. Cari SEMUA data prediksi yang merujuk ke profil pasien ini
        const predictions = await TbcarePrediction.find({ patient: patientProfile.user })
            .sort({ createdAt: -1 }); // Urutkan dari yang terbaru

        res.status(200).json({
            message: `Berhasil mengambil ${predictions.length} data riwayat untuk pasien ${participantId}.`,
            history: predictions
        });

    } catch (error) {
        console.error('ERROR di /api/v2/patient-history:', error);
        res.status(500).json({ message: 'Gagal mengambil data history.' });
    }
};

exports.updatePrediction = async (req, res, next) => {
    try {
        const { predictionId } = req.params; // Ambil ID prediksi dari URL
        const { newParticipantId, sputumStatus, sputumLevel } = req.body;

        // 1. Cari data prediksi yang mau diubah
        const prediction = await TbcarePrediction.findById(predictionId);
        if (!prediction) {
            return res.status(404).json({ message: 'Data prediksi tidak ditemukan.' });
        }

        // 2. Logika untuk mengubah mapping pasien (jika ada)
        if (newParticipantId) {
            const newPatientProfile = await TbcareProfile.findOne({ participantId: newParticipantId });
            if (!newPatientProfile) {
                return res.status(404).json({ message: `Pasien baru dengan ID ${newParticipantId} tidak ditemukan.` });
            }
            // Ganti patient ID di data prediksi
            prediction.patient = newPatientProfile.user;
        }
        
        // 3. Logika untuk mengubah sputum condition (jika ada)
        if (sputumStatus) {
            let sputumConditionLabel = sputumStatus;
            if (sputumStatus === 'Sputum +' && sputumLevel) {
                sputumConditionLabel += ` (${sputumLevel})`;
            }
            prediction.sputumCondition = sputumConditionLabel;
        }

        // 4. Simpan perubahan ke database
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


// === FUNGSI BARU UNTUK DELETE PREDIKSI ===
exports.deletePrediction = async (req, res, next) => {
    try {
        const { predictionId } = req.params; // Ambil ID prediksi dari URL

        // Cari dan hapus data prediksi berdasarkan ID-nya
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