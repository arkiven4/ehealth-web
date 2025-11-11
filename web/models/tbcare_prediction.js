const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const predictionSchema = new Schema(
  {
    patient: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    predictedBy: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    audioFile: {
      type: String,
      required: true,
    },
    sputumCondition: {
      type: String,
      required: false, // Changed to optional for feature flag
    },
    result: {
      type: String,
      required: true,
    },
    confidence: {
      type: Number,
      default: 0,
    },
    tbSegmentCount: { type: Number, default: 0 },
    nonTbSegmentCount: { type: Number, default: 0 },
    totalCoughSegments: { type: Number, default: 0 },
    // Validation status fields
    validationStatus: {
      type: String,
      enum: ["pending", "accepted", "rejected"],
      default: "pending",
    },
    validatedAt: {
      type: Date,
      default: null,
    },
    validatedBy: {
      type: Schema.Types.ObjectId,
      ref: "User",
      default: null,
    },
    validationNote: {
      type: String,
      default: null,
    },
    detail: {
      type: Schema.Types.Mixed,
      default: {},
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Prediction", predictionSchema);
