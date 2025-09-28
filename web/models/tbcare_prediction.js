const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const predictionSchema = new Schema({
    patient: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    predictedBy: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    audioFile: {
        type: String,
        required: true
    },

    sputumCondition: {
        type: String,
        required: true
    },

    result: {
        type: String,
        required: true
    },
    tbSegmentCount: { type: Number, default: 0 },
    nonTbSegmentCount: { type: Number, default: 0 },
    totalCoughSegments: { type: Number, default: 0 }
}, { timestamps: true });

module.exports = mongoose.model('Prediction', predictionSchema);