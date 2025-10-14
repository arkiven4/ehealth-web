const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const tbcareProfileSchema = new Schema({
    user: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    province: { type: String },
    city: { type: String },
    district: { type: String },
    sex: { type: String, enum: ['Male', 'Female'] },
    dateOfBirth: { type: Date },
    age: { type: Number },
    height: { type: Number },
    weight: { type: Number },
    bmi: { type: Number },
    weightStatus: { type: String },
    isCoughProductive: { type: String, enum: ['Yes', 'No'] },
    coughDurationDays: { type: Number },
    hasHemoptysis: { type: String, enum: ['Yes', 'No'] },
    hasChestPain: { type: String, enum: ['Yes', 'No'] },
    hasShortBreath: { type: String, enum: ['Yes', 'No'] },
    hasFever: { type: String, enum: ['Yes', 'No'] },
    hasNightSweats: { type: String, enum: ['Yes', 'No'] },
    hasWeightLoss: { type: String, enum: ['Yes', 'No'] },
    weightLossAmountKg: { type: Number },
    tobaccoUse: { type: String, enum: ['stopped', 'current', 'never', 'not disclosed'] },
    cigarettesPerDay: { type: Number },
    smokingSinceMonths: { type: Number },
    stoppedSmokingMonths: { type: Number },
    hadPriorTB: { type: String, enum: ['Yes', 'No'] },
    comorbidities: [{ type: String }],
    comorbiditiesOther: { type: String },
}, { timestamps: true });

module.exports = mongoose.model('TbcareProfile', tbcareProfileSchema);