
/*
************************************************************************************************************************
*
************************************************************************************************************************
*/

#ifndef UTILS_H
#define UTILS_H


/*
************************************************************************************************************************
*           INCLUDE FILES
************************************************************************************************************************
*/

#include <stdint.h>
#include "config.h"


/*
************************************************************************************************************************
*           DO NOT CHANGE THESE DEFINES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           CONFIGURATION DEFINES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           DATA TYPES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           GLOBAL VARIABLES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           MACRO'S
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           FUNCTION PROTOTYPES
************************************************************************************************************************
*/

uint32_t FrequencyToFFTBin(float frequency, uint32_t numBins, uint32_t sampleRate)

float FFTBinToFrequency(uint32_t bin, uint32_t numBins, uint32_t sampleRate)

float DegreesToRadians(float degrees)

float RadiansToDegrees(float radians)

float AmplitudeToDB(float volume)

float DBToAmplitude(float dB)

float SamplesToSeconds(uint32_t sample_rate, uint32_t samples)

uint32_t SecondsToSamples(uint32_t sample_rate, float seconds)

uint32_t MilliSecondsToSamples(uint32_t sample_rate, float milliseconds)

float SecondsToMilliseconds(float seconds)

float Frequency(float octave, float note)

void EnvelopeSamples(float *samples, uint32_t frame_size, float envelopeTimeFrontBack)

void NormalizeSamples(float *samples, uint32_t frame_size, float maxAmplitude)

void SineWave(float *frequencies, uint32_t bin, float startingPhase)

void SawWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)

void SquareWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)

void TriangleWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)

void NoiseWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)

/*
************************************************************************************************************************
*           CONFIGURATION ERRORS
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           END HEADER
************************************************************************************************************************
*/

#endif
