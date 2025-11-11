const fs = require("fs");
const path = require("path");
const { PythonShell } = require("python-shell");
const { spawn } = require("child_process");

const User = require("../models/user");
const TbcarePrediction = require("../models/tbcare_prediction");
const TbcareProfile = require("../models/tbcare_profile");
const Device_Data_TBPrimer = require("../models/device_data_cough_tbprimer");
const tbcareFeatures = require("../config/tbcare-features");

exports.getLandingPage = (req, res, next) => {
  res.render("tbcare/landing", { pageTitle: "Welcome to TBCare", isAuthenticated: req.session.isLoggedIn });
};
exports.getDownloadPage = (req, res, next) => {
  res.render("tbcare/download", { pageTitle: "Download TBCare App", isAuthenticated: req.session.isLoggedIn });
};
exports.getAboutPage = (req, res, next) => {
  res.render("tbcare/about", { pageTitle: "About TBCare", isAuthenticated: req.session.isLoggedIn });
};
exports.getTbcareLoginPage = (req, res, next) => {
  res.render("tbcare/login", { pageTitle: "TBCare Login", csrfToken: req.csrfToken() });
};

/**
 * @description Menampilkan halaman untuk memulai prediksi baru (menampilkan semua file audio).
 */
exports.getPredict = async (req, res, next) => {
  try {
    const allPatients = await User.find({
      role: "patient",
      doctor: req.session.user._id,
      tbcareProfile: { $exists: true, $ne: null },
    }).populate("tbcareProfile");

    const patients = allPatients.filter((patient) => patient.tbcareProfile && patient.tbcareProfile !== null);

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction",
      pageHeader: "Cough Recording Prediction",
      userdata: req.session.user,
      patients: patients,
      coughFiles: [], // <-- Diubah di sini: Langsung gunakan 'allFiles' tanpa difilter
      audioFolders: [],
      csrfToken: req.csrfToken(),
      errorMessage: req.flash("error")[0],
      enableSputumFields: tbcareFeatures.ENABLE_SPUTUM_FIELDS,
    });
  } catch (error) {
    console.log("Error in getPredict:", error);
    next(error);
  }
};

exports.getPredict_filteredWav = async (req, res, next) => {
  try {
    const nik = req.params.nik || req.query.nik;
    if (!nik) {
      return res.status(400).json({
        error: "NIK parameter is required",
        message: "Please provide a NIK parameter in the URL or query string",
      });
    }

    const batukData = await Device_Data_TBPrimer.find({ "json_data.nik": nik }, ["device_id", "cough_type", "time", "json_data"]);

    return res.json({
      success: true,
      nik: nik,
      count: batukData.length,
      data: batukData,
    });
  } catch (error) {
    console.log("Error in getPredict_filteredWav:", error);
    return res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
};

/**
 * @description (VERSI FINAL) Memproses form prediksi, menjalankan Python, menyimpan riwayat, dan menampilkan hasilnya.
 */
exports.postPredict = async (req, res, next) => {
  const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

  // Sputum is optional now (controlled by feature flag)
  if (!patientId || !coughFilePath) {
    req.flash("error", "Harap lengkapi semua kolom yang diperlukan.");
    return res.redirect("/tbcare/predict");
  }

  // Find patient by NIK (patientId is actually NIK from form)
  let patient;
  try {
    // First find TbcareProfile by NIK (NIK is stored as Number)
    const nikNumber = parseInt(patientId, 10);
    console.log("Searching for NIK:", nikNumber, "Type:", typeof nikNumber);

    // Try to find with Number first, then String if not found
    let tbcareProfile = await TbcareProfile.findOne({ nik: nikNumber });

    // If not found as Number, try as String
    if (!tbcareProfile) {
      console.log("Not found as Number, trying as String:", patientId);
      tbcareProfile = await TbcareProfile.findOne({ nik: patientId });
    }

    if (!tbcareProfile) {
      console.log("TbcareProfile not found. Checking all profiles with NIK field...");
      const allProfiles = await TbcareProfile.find({}, "nik").limit(5);
      console.log(
        "Sample NIKs in database:",
        allProfiles.map((p) => ({ nik: p.nik, type: typeof p.nik }))
      );
      req.flash("error", `Pasien dengan NIK ${patientId} tidak ditemukan.`);
      return res.redirect("/tbcare/predict");
    }

    console.log("Found TbcareProfile:", tbcareProfile._id);

    // Then find User by tbcareProfile reference
    patient = await User.findOne({ tbcareProfile: tbcareProfile._id }).populate("tbcareProfile");

    if (!patient) {
      req.flash("error", `Data pasien dengan NIK ${patientId} tidak ditemukan.`);
      return res.redirect("/tbcare/predict");
    }
  } catch (err) {
    console.error("Error finding patient:", err);
    req.flash("error", "Terjadi kesalahan saat mencari data pasien.");
    return res.redirect("/tbcare/predict");
  }

  // Normalize coughFilePath in case it starts with a leading slash (e.g. '/uploads/...')
  const normalizedCoughFile = (coughFilePath || "").replace(/^\/+/, "");
  const fileSystemPath = path.join(__dirname, "..", "public", normalizedCoughFile);
  if (!fs.existsSync(fileSystemPath)) {
    // provide a clearer debug message including attempted filesystem path
    req.flash("error", `File audio tidak ditemukan di server: ${fileSystemPath}`);
    return res.redirect("/tbcare/predict");
  }

  // replace the Python run + parse block with this safer helper + usage
  async function runPythonScript(pythonPath, args = []) {
    return new Promise((resolve, reject) => {
      const py = spawn("python3", [pythonPath, ...args], { stdio: ["ignore", "pipe", "pipe"] });
      let out = "";
      let err = "";

      py.stdout.on("data", (d) => {
        out += d.toString();
      });
      py.stderr.on("data", (d) => {
        err += d.toString();
      });

      py.on("close", (code) => {
        // Check if there's actual output first
        const txt = (out || "").trim();

        // Only reject if there's stderr AND no stdout AND exit code is non-zero
        if (!txt && err && code !== 0) {
          return reject(new Error(`Python script failed with exit code ${code}: ${err}`));
        }

        // Python script outputs multiple JSON lines (info messages + final result)
        // We need to parse the LAST JSON object which contains the prediction result
        let parsed;
        try {
          if (!txt) {
            return reject(new Error("Python script returned empty output"));
          }

          // Split by newlines and find all valid JSON objects
          const lines = txt.split(/\r?\n/).filter((l) => l.trim());
          let lastValidJson = null;

          for (let i = lines.length - 1; i >= 0; i--) {
            try {
              const jsonObj = JSON.parse(lines[i]);
              // Look for the final result (has status: "success" or "error" and prediction field)
              if (jsonObj.status && (jsonObj.prediction || jsonObj.message)) {
                lastValidJson = jsonObj;
                break;
              }
            } catch (e) {
              // Skip non-JSON lines
              continue;
            }
          }

          if (lastValidJson) {
            parsed = lastValidJson;
          } else {
            // Fallback: try to parse entire output as single JSON
            parsed = JSON.parse(txt);
          }
        } catch (e) {
          // Log full output for debugging
          console.error("Failed to parse Python output:");
          console.error("stdout:", out);
          console.error("stderr:", err);
          return reject(new Error(`Failed to parse python output. Parse error: ${e.message}`));
        }

        resolve({ parsed, stdout: out, stderr: err, code });
      });

      py.on("error", (e) => {
        reject(e);
      });
    });
  }

  try {
    const pythonScriptPath = path.join(__dirname, "..", "python-script", "tbcareScript.py");
    const audioPathForPython = fileSystemPath;

    console.log("=== TB Care Prediction Debug ===");
    console.log("Patient NIK:", patientId);
    console.log("Patient _id:", patient._id);
    console.log("Audio file path:", coughFilePath);
    console.log("Normalized file:", normalizedCoughFile);
    console.log("Filesystem path:", fileSystemPath);
    console.log("Python script path:", pythonScriptPath);
    console.log("Audio path for Python:", audioPathForPython);

    const result = await runPythonScript(pythonScriptPath, [audioPathForPython]);

    console.log("Python stdout length:", result.stdout?.length || 0);
    console.log("Python stderr length:", result.stderr?.length || 0);
    console.log("Python exit code:", result.code);
    const parsed = result.parsed;

    if (!parsed || parsed.status === "error") {
      console.error("Python returned error or empty output");
      console.error("Parsed result:", parsed);
      console.error("Full stdout:", result.stdout);
      console.error("Full stderr:", result.stderr);
      const errorMsg = parsed?.message || "Terjadi kesalahan sistem saat menjalankan skrip analisis.";
      req.flash("error", errorMsg);
      return res.redirect("/tbcare/predict");
    }

    // Check if prediction was successful
    if (parsed.status !== "success") {
      console.error("Python script status not success:", parsed);
      req.flash("error", parsed.message || "Gagal memproses audio.");
      return res.redirect("/tbcare/predict");
    }

    // Extract new format fields
    const finalDecision = parsed.prediction || parsed.final_decision || "UNKNOWN";
    const detail = parsed.detail || {};
    const segmentDetails = parsed.segment_details || [];

    const totalSegments = detail.total_segments || 0;
    const validCoughSegments = detail.valid_cough_segments || 0;
    const tbSegments = detail.tb_segments || 0;
    const nonTbSegments = detail.non_tb_segments || 0;
    const avgTbProbability = detail.average_tb_probability || 0;
    const confidencePercentage = detail.confidence_percentage || 0;

    // Save prediction to database (patient already fetched at the beginning)
    const predictionData = {
      patient: patient._id,
      predictedBy: req.session.user._id,
      audioFile: coughFilePath,
      result: finalDecision,
      confidence: avgTbProbability,
      tbSegmentCount: tbSegments,
      nonTbSegmentCount: nonTbSegments,
      totalCoughSegments: validCoughSegments,
      detail: {
        total_segments: totalSegments,
        valid_cough_segments: validCoughSegments,
        average_tb_probability: avgTbProbability,
        confidence_percentage: confidencePercentage,
        segment_details: segmentDetails,
        waveform: parsed.waveform,
        mfcc: parsed.mfcc,
      },
    };

    // Add sputum if provided (feature flag controlled)
    if (sputumStatus) {
      predictionData.sputumCondition = sputumStatus;
    }

    const newPrediction = new TbcarePrediction(predictionData);
    await newPrediction.save();

    // Render result page with new data
    res.render("doctor/tbcare/predict-result", {
      pageTitle: "Hasil Prediksi TB",
      pageHeader: "Hasil Prediksi Tuberkulosis",
      userdata: req.session.user,
      csrfToken: req.csrfToken(),
      patient: patient,
      prediction: finalDecision,
      detail: detail,
      segmentDetails: segmentDetails,
      sputumCondition: sputumStatus,
      audioFile: coughFilePath,
      waveform: JSON.stringify(parsed.waveform || []),
      mfcc: JSON.stringify(parsed.mfcc || []),
      totalSegments: totalSegments,
      validCoughSegments: validCoughSegments,
      tbSegments: tbSegments,
      nonTbSegments: nonTbSegments,
      avgTbProbability: avgTbProbability,
      confidencePercentage: confidencePercentage,
      // Validation data (for new predictions, not yet validated)
      predictionId: newPrediction._id,
      validationStatus: "pending", // New predictions start as pending
      predictionDate: newPrediction.createdAt,
    });
  } catch (err) {
    console.error("=== TB Care Prediction Error ===");
    console.error("Error message:", err.message || err);
    console.error("Error stack:", err.stack);
    console.error("Full error object:", err);
    req.flash("error", `Terjadi kesalahan sistem: ${err.message || "Unknown error"}`);
    return res.redirect("/tbcare/predict");
  }
};

exports.getPatientHistoryList = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate("tbcareProfile");

    // Fetch last prediction for each patient
    const patientsWithLastPrediction = await Promise.all(
      patients.map(async (patient) => {
        const lastPrediction = await TbcarePrediction.findOne({ patient: patient._id }).sort({ createdAt: -1 }).limit(1);
        const patientObj = patient.toObject();

        // Debug: Log patient data
        console.log("Patient:", patientObj.userName, "Sex:", patientObj.tbcareProfile?.sex);

        return {
          ...patientObj,
          lastPrediction: lastPrediction,
        };
      })
    );

    res.render("doctor/tbcare/patient-history-list", {
      pageTitle: "Patient History",
      pageHeader: "Patient Prediction History",
      userdata: req.session.user,
      patients: patientsWithLastPrediction,
      csrfToken: req.csrfToken(),
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
    const patient = await User.findById(patientId).populate("tbcareProfile");

    if (!patient) {
      return res.redirect("/tbcare/patient-history");
    }

    const predictions = await TbcarePrediction.find({ patient: patientId }).populate("predictedBy", "userName").sort({ createdAt: -1 });

    res.render("doctor/tbcare/patient-history-detail", {
      pageTitle: `History for ${patient.fullName.first}`,
      pageHeader: "Prediction History",
      userdata: req.session.user,
      patient: patient,
      predictions: predictions,
      csrfToken: req.csrfToken(),
      enableSputumFields: tbcareFeatures.ENABLE_SPUTUM_FIELDS,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @description Menampilkan detail lengkap hasil prediksi dari history (seperti predict-result)
 */
exports.getPredictionDetail = async (req, res, next) => {
  try {
    const predictionId = req.params.predictionId;
    const prediction = await TbcarePrediction.findById(predictionId).populate("patient").populate("predictedBy", "userName");

    if (!prediction) {
      req.flash("error", "Prediction not found.");
      return res.redirect("/tbcare/patient-history");
    }

    // Extract data from prediction.detail
    const detail = prediction.detail || {};
    const segmentDetails = detail.segment_details || [];
    const waveform = detail.waveform || [];
    const mfcc = detail.mfcc || [];

    // Calculate confidence properly
    let confidenceValue = prediction.confidence || 0;
    let avgTbProb = prediction.confidence || 0;

    // If confidence is not saved, calculate from detail
    if (!confidenceValue && detail.average_tb_probability) {
      avgTbProb = detail.average_tb_probability;
      confidenceValue = avgTbProb;
    }

    // If still no confidence, estimate from TB segments ratio (fallback for old data)
    if (!confidenceValue && prediction.totalCoughSegments > 0) {
      const ratio = prediction.tbSegmentCount / prediction.totalCoughSegments;
      confidenceValue = ratio;
      avgTbProb = ratio;
      console.log("Using fallback confidence calculation from segments ratio:", ratio);
    }

    // Ensure confidence is a valid number
    if (isNaN(confidenceValue)) confidenceValue = 0;
    if (isNaN(avgTbProb)) avgTbProb = 0;

    const confidencePercentage = parseFloat((confidenceValue * 100).toFixed(2));

    console.log("History View Debug:", {
      predictionId: prediction._id,
      confidence: prediction.confidence,
      confidencePercentage: confidencePercentage,
      hasDetail: !!detail,
      hasWaveform: waveform.length > 0,
      hasMfcc: mfcc.length > 0,
      segmentCount: segmentDetails.length,
    });

    // Render using the same view as predict-result
    res.render("doctor/tbcare/predict-result", {
      pageTitle: "Prediction History Detail",
      pageHeader: "Historical Prediction Result",
      userdata: req.session.user,
      patient: prediction.patient,
      prediction: prediction.result,
      audioFile: prediction.audioFile,
      sputumCondition: prediction.sputumCondition,
      confidencePercentage: confidencePercentage,
      avgTbProbability: avgTbProb,
      totalSegments: detail.total_segments || 0,
      validCoughSegments: prediction.totalCoughSegments || detail.valid_cough_segments || 0,
      tbSegments: prediction.tbSegmentCount || detail.tb_segments || 0,
      nonTbSegments: prediction.nonTbSegmentCount || detail.non_tb_segments || 0,
      segmentDetails: segmentDetails,
      waveform: JSON.stringify(waveform),
      mfcc: JSON.stringify(mfcc),
      csrfToken: req.csrfToken(),
      isHistoryView: true, // Flag to show "Back to History" button instead
      predictionDate: prediction.createdAt,
      // Validation data (for history view)
      predictionId: prediction._id,
      validationStatus: prediction.validationStatus || "pending",
      validatedAt: prediction.validatedAt,
      validatedBy: prediction.validatedBy,
      validationNote: prediction.validationNote,
    });
  } catch (error) {
    console.error("Error fetching prediction detail:", error);
    next(error);
  }
};
