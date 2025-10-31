const fs = require("fs");
const path = require("path");
const { PythonShell } = require("python-shell");
const { spawn } = require("child_process");

const User = require("../models/user");
const TbcarePrediction = require("../models/tbcare_prediction");
const TbcareProfile = require("../models/tbcare_profile");
const Device_Data_TBPrimer = require("../models/device_data_cough_tbprimer");

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
      tbcareProfile: { $exists: true, $ne: null}
    }).populate("tbcareProfile");

    const patients = allPatients.filter(patient => 
      patient.tbcareProfile && patient.tbcareProfile !== null
    );

    res.render("doctor/tbcare/predict", {
      pageTitle: "TBCare - Cough Prediction",
      pageHeader: "Cough Recording Prediction",
      userdata: req.session.user,
      patients: patients,
      coughFiles: [], // <-- Diubah di sini: Langsung gunakan 'allFiles' tanpa difilter
      audioFolders: [],
      csrfToken: req.csrfToken(),
      errorMessage: req.flash("error")[0],
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
        error: 'NIK parameter is required',
        message: 'Please provide a NIK parameter in the URL or query string'
      });
    }

    const batukData = await Device_Data_TBPrimer.find(
      { 'json_data.nik': nik },
      ['device_id', 'cough_type', 'time', 'json_data']
    );

    return res.json({
      success: true,
      nik: nik,
      count: batukData.length,
      data: batukData
    });
  } catch (error) {
    console.log("Error in getPredict_filteredWav:", error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
};

/**
 * @description (VERSI FINAL) Memproses form prediksi, menjalankan Python, menyimpan riwayat, dan menampilkan hasilnya.
 */
exports.postPredict = async (req, res, next) => {
  const { patientId, coughFilePath, sputumStatus, sputumLevel } = req.body;

  if (!patientId || !coughFilePath || !sputumStatus) {
    req.flash("error", "Harap lengkapi semua kolom yang diperlukan.");
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
        if (err && !out) {
          // prefer predict output but surface stderr for debugging
          return reject(new Error(`Python stderr: ${err}`));
        }

        // try parse JSON first, otherwise try CSV/line format
        let parsed;
        try {
          parsed = out && out.trim() ? JSON.parse(out) : null;
        } catch (e) {
          // fallback: try to parse simple comma separated values or key=value pairs
          try {
            const txt = (out || "").trim();
            if (!txt) throw e;
            // attempt CSV -> array
            if (txt.indexOf(",") !== -1) {
              parsed = txt.split(",").map((s) => s.trim());
            } else {
              // key:value lines
              parsed = {};
              txt.split(/\r?\n/).forEach((line) => {
                const m = line.match(/^\s*([^:=]+)\s*[:=]\s*(.+)$/);
                if (m) parsed[m[1].trim()] = m[2].trim();
              });
            }
          } catch (e2) {
            // keep original parse error but return full output for debugging
            return reject(new Error(`Failed to parse python output. stdout:${out}\nstderr:${err}\nparseError:${e.message}`));
          }
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
    const result = await runPythonScript(pythonScriptPath, [fileSystemPath]);

    // result.parsed may be object, array or null
    const parsed = result.parsed;

    if (!parsed) {
      console.error("Python returned empty output", result);
      req.flash("error", "Terjadi kesalahan sistem saat menjalankan skrip analisis (empty output).");
      return res.redirect("/tbcare/predict"); // or respond with 500 JSON for API
    }

    // defensive extraction of fields
    const totalSegments = parsed.total_segments ?? parsed.totalSegments ?? parsed.total ?? null;
    if (totalSegments === null) {
      console.error("Missing total segments in python output", result);
      req.flash("error", `Analisis gagal — output skrip tidak berisi informasi segmen. raw:${result.stdout}`);
      return res.redirect("/tbcare/predict");
    }

    // get other fields defensively (example)
    const finalDecision = parsed.final_decision ?? parsed.result ?? (Array.isArray(parsed) ? parsed[0] : null);
    const tbSegments = parsed.tb_segments ?? parsed.tbSegmentCount ?? parsed.tbSegment ?? 0;
    const nonTbSegments = parsed.non_tb_segments ?? parsed.nonTbSegmentCount ?? parsed.nonTbSegment ?? 0;

    const details = parsed.detail;
    const confidenceScore = totalSegments > 0 ? tbSegments / totalSegments : 0;

    const newPrediction = new TbcarePrediction({
      patient: patientId,
      predictedBy: req.session.user._id,
      audioFile: coughFilePath,
      sputumCondition: sputumStatus,
      result: finalDecision,
      confidence: confidenceScore,
      tbSegmentCount: tbSegments,
      nonTbSegmentCount: nonTbSegments,
      totalCoughSegments: totalSegments,
      detail: { ...details, waveform: parsed.waveform, mfcc: parsed.mfcc },
    });
    await newPrediction.save();

    res.render("doctor/tbcare/predict-result", {
      pageTitle: "Hasil Prediksi",
      pageHeader: "Hasil Prediksi",
      userdata: req.session.user,
      csrfToken: req.csrfToken(),
      patient: patient,
      prediction: finalDecision,
      detail: details,
      sputumCondition: sputumStatus,
      audioFile: coughFilePath,
      waveform: JSON.stringify(parsed.waveform),
      mfcc: JSON.stringify(parsed.mfcc),
    });
  } catch (err) {
    console.error("Failed to parse JSON from Python script:", err.message || err);
    console.error("Full error:", err);
    // show helpful message to user and preserve stderr/stdout where available
    req.flash("error", "Terjadi kesalahan sistem saat menjalankan skrip analisis.");
    return res.redirect("/tbcare/predict");
  }
};

exports.getPatientHistoryList = async (req, res, next) => {
  try {
    const patients = await User.find({ role: "patient", doctor: req.session.user._id }).populate("tbcareProfile");
    res.render("doctor/tbcare/patient-history-list", {
      pageTitle: "Patient History",
      pageHeader: "Patient Prediction History",
      userdata: req.session.user,
      patients: patients,
      csrfToken: req.csrfToken(), // <-- TAMBAHKAN BARIS INI
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
    });
  } catch (error) {
    next(error);
  }
};
