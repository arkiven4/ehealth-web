const bodyParser = require("body-parser");
const multer = require("multer");

const imageStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "public/data/images");
  },
  filename: (req, file, cb) => {
    cb(null, new Date().toISOString() + "-" + file.originalname);
  },
});

const exelStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "exels");
  },
  filename: (req, file, cb) => {
    cb(null, new Date().toISOString() + "-" + file.originalname);
  },
});

const imageFilter = (req, file, cb) => {
  if (
    file.mimetype === "image/png" ||
    file.mimetype === "image/jpg" ||
    file.mimetype === "image/jpeg"
  ) {
    cb(null, true);
  } else {
    cb(null, false);
  }
};

const excelFilter = (req, file, cb) => {
  if (
    file.mimetype.includes("excel") ||
    file.mimetype.includes("spreadsheetml")
  ) {
    cb(null, true);
  } else {
    cb(null, false);
  }
};

exports.imageUploadHandler = multer({
  storage: imageStorage,
  fileFilter: imageFilter,
}).array("image");
exports.excelFilterUploadHandler = multer({
  storage: exelStorage,
  fileFilter: excelFilter,
}).single("exel");
exports.bodyParserHandler = bodyParser.urlencoded({ extended: false });
exports.bodyJsonHandler = bodyParser.json();

const audioStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "public/uploads/tbcare");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname.replace(/\s/g, '_'));
  },
});

const audioFilter = (req, file, cb) => {
  if (
    file.mimetype === "audio/wav" ||
    file.mimetype === "audio/mpeg" ||
    file.mimetype === "audio/ogg"
  ) {
    cb(null, true);
  } else {
    cb(new Error("Hanya file audio (.wav, .mp3, .ogg) yang diizinkan!"), false);
  }
};

exports.bodyParserHandler = bodyParser.urlencoded({ extended: false });
exports.bodyJsonHandler = bodyParser.json();

exports.audioUploadHandler = multer({
  storage: audioStorage,
  fileFilter: audioFilter,
});

exports.imageUploadHandler = multer({
  storage: imageStorage,
  fileFilter: imageFilter,
}).array("image");
exports.excelFilterUploadHandler = multer({
  storage: exelStorage,
  fileFilter: excelFilter,
}).single("exel");